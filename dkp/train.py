from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch
import torch.nn as nn

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
    nets = DenoiseTransformerNet(img_size=args.image_size)
    nets = nets.to(rank)
    ddp_net = DDP(nets, device_ids=[rank])

    if rank == 0:
        # only need one ema model
        ema_model = EMAModel(ddp_net.parameters(), power=0.75)
    
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
    
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
    
    
    # writter
    if rank == 0:
        
        writer = SummaryWriter(f'logs/{args.name}')
        losses = deque(maxlen=args.log_interval)
        best_loss = 1e10
        
    def eval(net, step):
        """
        Define a metric
        """
        
        # for now, simply sample a traj_map from test dataset, sample kp using trained net, and compare to gt kp
        
        tra_map, gt_kp = test_dataset[0]
        tra_map = tra_map.unsqueeze(0).to(rank) # [1,1,H,W]

        n_sample = 200
        cond = net.module.get_vis_feat(tra_map) # [ batch, feat_dim]
        
        # noise kps
        noisy_kp = torch.rand(n_sample, 2).to(rank) * 2 - 1
        noisy_kp = noisy_kp.unsqueeze(0) # [1, n_sample, 2]
        
        noise_scheduler.set_timesteps(args.num_train_timesteps)  
        
        for k in noise_scheduler.timesteps:
            # ts = torch.ones(n_sample).long().to(rank) * k
            ts = torch.tensor([k]).long().to(rank)
            noise_pred = net.module.get_noise_pred(noisy_kp, cond, ts)
            noisy_kp = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_kp,
            ).prev_sample
            
        # unnormalize and plot
        pred_kp = torch.clamp(noisy_kp, -1, 1)
        pred_kp = pred_kp * (args.image_size/2) + (args.image_size/2)
        pred_kp = pred_kp.clamp(0, args.image_size-1)
        pred_kp = pred_kp.cpu().detach().numpy()
        pred_kp = pred_kp.astype(np.int32)
        pred_kp = pred_kp[0] # [n_sample, 2]
        
        # save pred_kp
        base_map = tra_map.clone()
        base_map[tra_map==0] = 255 # obstacle
        base_map[tra_map==1] = 0 # traversible
        
        m = base_map.cpu().numpy().astype(np.uint8)[0,0]
        for k in pred_kp:
            m[k[0], k[1]] = 128
        writer.add_image('eval/pred_kp', m, step, dataformats='HW')
        
        
        # unnormalize gt_kp
        gt_kp = gt_kp.numpy()
        gt_kp = gt_kp  * (args.image_size/2) + (args.image_size/2)
        gt_kp = np.clip(gt_kp, 0, args.image_size-1)
        gt_kp = gt_kp.astype(np.int32)
        
        # # save gt_kp
        m = base_map.cpu().numpy().astype(np.uint8)[0,0]
        for k in gt_kp:
            m[k[0], k[1]] = 128
        writer.add_image('eval/gt_kp', m, step,dataformats='HW')
                    
            
    for epoch in range(args.num_epochs):
        if rank == 0:
            dataloader = tqdm(dataloader)
            
        for i, batch in enumerate(dataloader):
            tra_map, kp = batch
            tra_map = tra_map.to(rank)
            kp = kp.to(rank)
            bs = kp.shape[0]
            
            # sample timesteps
            timesteps = torch.randint(0, args.num_train_timesteps, (bs,)).long().to(rank)

            # sample noisy kps
            noise = torch.randn_like(kp)            
            noisy_kp = noise_scheduler.add_noise(kp,noise,timesteps)
            
            # noise_pred = ddp_net['noise_pred'](noisy_kp, cond, timesteps)
            noise_pred = ddp_net(tra_map, noisy_kp, timesteps)
            
            loss = torch.mean((noise_pred - noise)**2)
            
            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            if rank == 0:
                # update ema model
                losses.append(loss.item())
                ema_model.step(ddp_net.parameters())
                
            # log and save
                if i % args.log_interval == 0:
                    dataloader.set_description(f'Epoch {epoch} | Loss {np.mean(losses)}')
                    writer.add_scalar('train/loss', np.mean(losses), epoch * len(dataloader) + i)
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
    parser.add_argument("--master_port", type=str, help="port for master", default='6667')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_train_timesteps", type=int, default=100) # denoising steps
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ema_decay", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
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