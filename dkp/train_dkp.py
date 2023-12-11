from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import VisualEncoder, MLPNoisePredNet, MLPDenoise, DenoiseTransformerNet
from .kp_dataset import KPDataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import argparse
from collections import deque
import os
import matplotlib.pyplot as plt
import imageio

from skimage.draw import line,line_aa
def cleanup():
    dist.destroy_process_group()
    
def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_process(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)

    set_seed(args.seed + rank)
    run(rank, size, args)    

    dist.barrier()
    cleanup()


def draw_kp(kp, base_map, writer=None, step=None, name=None,unnormalize=True):
    m = base_map.cpu().numpy().astype(np.uint8)[0,0]
    image_size = m.shape[0]
    if unnormalize:
        pred_kp = torch.clamp(kp, -1, 1)
        pred_kp = pred_kp * (image_size/2) + (image_size/2)
        pred_kp = pred_kp.clamp(0, image_size-1)
        pred_kp = pred_kp.cpu().detach().numpy()
        pred_kp = pred_kp.astype(np.int32)
        pred_kp = pred_kp.squeeze(0) # [n_sample, 2]

    m[pred_kp[:,0], pred_kp[:,1]] = 128
    
    if writer is not None:
        writer.add_image(name, m, step, dataformats='HW')

    return m

def draw_lines(lines, base_map, writer=None, step=None, name=None, unnormalize=True,logits=None):
    m = base_map.cpu().numpy().astype(np.uint8)[0,0]
    m = np.stack([m,m,m], axis=-1)
    colours = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]]) # red, green, blue, yellow, cyan
    
    image_size = m.shape[0]
    if unnormalize:
        lines = lines * (image_size/2) + (image_size/2)
        lines = lines.clamp(0, image_size-1)
    lines = lines.clamp(0, image_size-1)
    
    for i, line in enumerate(lines):
        line = line.cpu().detach().numpy()
        line = line.astype(np.int32)
        rr,cc, _ = line_aa(line[0], line[1], line[2], line[3])

        # each line has a different color
        m[rr, cc] = colours[i % len(colours)]

    # draw classification results
    if logits is not None:
        for i, l in enumerate(logits):
            if l > 0: # intersect
                x = i * 10
                y = 0
                m[x:x+10, y:y+10] = colours[i % len(colours)]
        
    if writer is not None:
        writer.add_image(name, m, step, dataformats='HWC')
    return m

def grad_lc_fn(x,cond, net):
    """ x: [2,2]"""
    n_line = x.shape[1] // 2

    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = net.module.get_line_pred(x_in.view(n_line,1,4), cond.repeat_interleave(n_line,dim=0)).view(-1)
        prob = 1 - torch.sigmoid(logits)
        log_prob = torch.log(prob)
        grad = torch.autograd.grad(log_prob.sum(), x_in)[0]
    
        
    return grad.view(1,n_line*2,2), logits

def grad_line_dist_fn(x):
    """ x: [2,2]"""
    n_line = x.shape[1] // 2

    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        p1s = x_in[:,:-1,:]
        p2s = x_in[:,1:,:]
        dist = torch.norm(p1s - p2s, dim=-1)
        
        grad = torch.autograd.grad(dist.sum(), x_in)[0]
    
        grad = grad.view(1,n_line*2,2)
        grad[:,0,:] = 0 # only update p2s
        grad[:,-1,:] = 0 # only update p2s

    return grad

def grad_traj_dist_fn(x):
    """ x: [2,2]"""
    n_line = x.shape[1] // 2

    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        p1s = x_in[:,1::2,:]
        p2s = x_in[:,2::2,:]
        dist = torch.norm(p1s - p2s, dim=-1)
        
        grad = torch.autograd.grad(dist.sum(), x_in)[0]
    
        grad = grad.view(1,n_line*2,2)
        grad[:,2::2,:] = 0 # only update p2s

    return grad
    
def run(rank,world_size, args):
    """
    Run DKP training on a single GPU 
    """
    
    dataset = KPDataset(split='train', rank=rank, world_size=world_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    if rank == 0:
        test_dataset = KPDataset(split='test', rank=rank, world_size=world_size)
        test_loader = torch.utils.data.DataLoader(  
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
    nets = DenoiseTransformerNet(img_size=args.image_size)
    nets = nets.to(rank)
    ddp_net = DDP(nets, device_ids=[rank])

    if rank == 0:
        # only need one ema model
        ema_model = EMAModel(ddp_net.parameters(), power=0.75)
    
    if not args.use_adamw:
        optimizer = torch.optim.Adam(ddp_net.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
    else:
        optimizer = torch.optim.AdamW(ddp_net.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule='squaredcos_cap_v2', # better?
        clip_sample=True, # clip to [-1, 1]
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epochs,
    )
    
    lc_loss_fn = nn.BCEWithLogitsLoss()
    # writter
    if rank == 0:
        
        writer = SummaryWriter(f'logs/{args.name}')
        losses = deque(maxlen=args.log_interval)
        losses_di = deque(maxlen=args.log_interval)
        losses_lc = deque(maxlen=args.log_interval)
        best_loss = 1e10
        
    def eval(net, step):
        """
        Define a metric
        """
        
        ######################################
        # visualization
        ######################################
        
        map_idx = np.random.randint(0, len(test_dataset))
        tra_map, gt_kp, lines, labels = test_dataset[map_idx]
        tra_map = tra_map.unsqueeze(0).to(rank) # [1,1,H,W]

        base_map = tra_map.clone()
        base_map[tra_map==0] = 255 # obstacle
        base_map[tra_map==1] = 0 # traversible

        lines = lines.to(rank) # bs, N, 4
        labels = labels.to(rank) # bs, N
        n_lines = lines.shape[0]
        n_sample = 200
        cond = net.module.get_vis_feat(tra_map) # [ batch, feat_dim]
        
        ######################################
        # kp denoising sample unconditioned
        ######################################
        # noise kps
        noisy_kp = torch.rand(n_sample, 2).to(rank) * 2 - 1
        noisy_kp = noisy_kp.unsqueeze(0) # [1, n_sample, 2]
        
        noise_scheduler.set_timesteps(args.num_train_timesteps)  
        traj = []
        for k in noise_scheduler.timesteps:
            # ts = torch.ones(n_sample).long().to(rank) * k
            ts = torch.tensor([k]).long().to(rank)
            noise_pred = net.module.get_noise_pred(noisy_kp, cond, ts)
            noisy_kp = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_kp,
            ).prev_sample
            if k % args.traj_save_int == 0:
                img = draw_kp(noisy_kp, base_map)
                traj.append(img)
            
        writer.add_images('eval/traj_kp', np.concatenate(traj,axis=1), step, dataformats='HW')
        draw_kp(noisy_kp, base_map, writer, step, 'eval/pred_kp')
        
        
        ######################################
        # kp denoising sample unconditioned
        ######################################
        #--------------------- classifier guided kp ---------------------#
        n_line = 5
        n_sample = n_line * 2
        noisy_kp = torch.rand(n_sample, 2).to(rank) * 2 - 1
        noisy_kp = noisy_kp.unsqueeze(0) # [1, n_sample, 2]
        
        noise_scheduler.set_timesteps(args.num_train_timesteps)  
        
        traj = []
        for k in noise_scheduler.timesteps:
            # ts = torch.ones(n_sample).long().to(rank) * k
            ts = torch.tensor([k]).long().to(rank)
            noise_pred = net.module.get_noise_pred(noisy_kp, cond, ts)
            grad_lc, logits = grad_lc_fn(noisy_kp, cond, net)
            noisy_kp = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_kp,
            ).prev_sample # [1, n_sample, 2]
            v = noise_scheduler._get_variance(k)

            # enforce small dist
            grad_dist = grad_line_dist_fn(noisy_kp)
            # enforce small between consecutive points
            
            noisy_kp = noisy_kp + grad_lc * v -  grad_dist * v
                        
            if k % args.traj_save_int == 0:
                lines = noisy_kp.view(n_line,4)
                # run classifier on lines
                lc_pred = net.module.get_line_pred(lines.view(-1,1,4), cond.repeat_interleave(n_line,dim=0)).view(-1)
                                
                img = draw_lines(lines, base_map, logits=lc_pred)
                traj.append(img)
            
        writer.add_images('eval/traj_kp_lc', np.concatenate(traj,axis=1), step, dataformats='HWC')
            
        lines = noisy_kp.view(n_line,4)
        draw_lines(lines, base_map, writer, step, 'eval/pred_kp_lc')
        
        #--------------------- gt kp ---------------------#
        draw_kp(gt_kp.unsqueeze(0), base_map, writer, step, 'eval/gt_kp')
        
        ######################################
        # line prediction accuracy
        ######################################
        total_correct = 0
        total = 0
        
        for batch in test_loader:
            tra_map, kp, lines, labels = batch
            tra_map = tra_map.to(rank)
            lines = lines.to(rank) # bs, N, 4
            labels = labels.to(rank) # bs, N
            cond = net.module.get_vis_feat(tra_map)
            n_lines = lines.shape[1]
            pred = net.module.get_line_pred(lines.view(-1,1,4), cond.repeat_interleave(n_lines,dim=0)).view(-1)
            total_correct += ((pred > 0) == labels.view(-1).float()).float().sum()
            total += len(pred)
        
            
        acc = total_correct / total
        writer.add_scalar('eval/lc_acc', acc, step)
            
    for epoch in range(args.num_epochs):
        if rank == 0:
            dataloader = tqdm(dataloader)
            
        for i, batch in enumerate(dataloader):
            tra_map, kp, lines, labels = batch
            tra_map = tra_map.to(rank)
            kp = kp.to(rank) # bs, N, 2
            lines = lines.to(rank) # bs, N, 4
            labels = labels.to(rank) # bs, N
            bs = kp.shape[0]
            
            ######################################
            # kp denoising loss
            ######################################
            # sample timesteps
            timesteps = torch.randint(0, args.num_train_timesteps, (bs,)).long().to(rank)

            # sample noisy kps
            noise = torch.randn_like(kp)            
            noisy_kp = noise_scheduler.add_noise(kp,noise,timesteps)
            
            # noise_pred = ddp_net['noise_pred'](noisy_kp, cond, timesteps)
            cond = ddp_net.module.get_vis_feat(tra_map)
            noise_pred = ddp_net.module.get_noise_pred(noisy_kp, cond, timesteps)
            # noise_pred = ddp_net(tra_map, noisy_kp, timesteps)
            
            loss_di = torch.mean((noise_pred - noise)**2)

            ######################################
            # line prediction loss
            ######################################
            line_pred = ddp_net.module.get_line_pred(lines.view(-1,1,4), cond.repeat_interleave(dataset.d_pairs_per_map,dim=0))
            loss_lc = lc_loss_fn(line_pred.view(-1), labels.view(-1).float())       
            

            loss = loss_di + 0.5 * loss_lc
            
            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            if rank == 0:
                # update ema model
                losses.append(loss.item())
                losses_di.append(loss_di.item())
                losses_lc.append(loss_lc.item())
                ema_model.step(ddp_net.parameters())
                
            # log and save
                if i % args.log_interval == 0:
                    dataloader.set_description(f'Epoch {epoch} | Loss {np.mean(losses)}')
                    writer.add_scalar('train/loss', np.mean(losses), epoch * len(dataloader) + i)
                    writer.add_scalar('train/loss_di', np.mean(losses_di), epoch * len(dataloader) + i)
                    writer.add_scalar('train/loss_lc', np.mean(losses_lc), epoch * len(dataloader) + i)
                    
                    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch * len(dataloader) + i)
                    if np.mean(losses) < best_loss:
                        best_loss = np.mean(losses)
                        torch.save(ddp_net.state_dict(), f'checkpoints/{args.name}.pth')
                        torch.save(ema_model.state_dict(), f'checkpoints/{args.name}_ema.pth')

                # eval
                if i % args.eval == 0:
                    with torch.no_grad():
                        ema_model.store(ddp_net.parameters())
                        ema_model.copy_to(ddp_net.parameters())
                        ddp_net.eval()
                        eval(ddp_net, epoch * len(dataloader) + i)
                        ema_model.restore(ddp_net.parameters())
                        ddp_net.train()
            
        if rank == 0:
            dataloader.close()

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_address", type=str, help="address for master", default='localhost')
    parser.add_argument("--master_port", type=str, help="port for master", default='6666')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_train_timesteps", type=int, default=50) # denoising steps
    parser.add_argument("--traj_save_int", type=int, default=5) # denoising steps
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ema_decay", type=float, default=0.75)
    parser.add_argument("--use_adamw", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--name", type=str, default='debug') 
    parser.add_argument("--num_proc_node", type=int, default=1) # number of processes per node
    parser.add_argument("--n_gpu_per_node", type=int, default=4) # number of gpus per node
    parser.add_argument("--gpu", type=int, default=0) # use single GPU
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=192)

    args = parser.parse_args()
    return args

def main():
    
    args = get_args()
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        init_process(0, 1, args)
        
    else:
        size = args.num_proc_node * args.n_gpu_per_node
        mp.spawn(init_process, args=(size, args), nprocs=size, join=True)