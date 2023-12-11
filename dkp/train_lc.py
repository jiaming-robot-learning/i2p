"""
line classifier
"""



from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch
import torch.nn as nn

from .net import VisualEncoder, MLPNoisePredNet, MLPDenoise, DenoiseTransformerNet, LineClassifier
from .kp_dataset import KPDataset, KPLDataset

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
    
    dataset = KPLDataset(split='train', rank=rank, world_size=world_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    if rank == 0:
        test_dataset = KPLDataset(split='test', rank=rank, world_size=world_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    nets = LineClassifier(192)
    nets = nets.to(rank)
    ddp_net = DDP(nets, device_ids=[rank])
    loss_fn = nn.BCEWithLogitsLoss()

    if rank == 0:
        # only need one ema model
        ema_model = EMAModel(ddp_net.parameters(), power=0.75)
    
    if not args.use_adamw:
        optimizer = torch.optim.Adam(ddp_net.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
    else:
        optimizer = torch.optim.AdamW(ddp_net.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
    


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
        total_correct = 0
        total = 0
        print('evaluating...')
        for batch in tqdm(test_dataloader):
            tra_map, l, label = batch
            tra_map = tra_map.to(rank)
            l = l.to(rank).unsqueeze(1)
            label = label.to(rank)
            
            # sample timesteps

            # sample noisy kps
            map_feat = net.module.map_encoder(tra_map)
            pred = net(l, map_feat).squeeze(1)

            total_correct += ((pred > 0) == label).sum().item()
            total += label.shape[0]
        acc = total_correct / total
        writer.add_scalar('test/acc', acc, step)
        print(f'Acc: {acc}')
        return acc
                         
            
    for epoch in range(args.num_epochs):
        if rank == 0:
            dataloader = tqdm(dataloader)
            
        for i, batch in enumerate(dataloader):
            tra_map, l, label = batch
            tra_map = tra_map.to(rank)
            l = l.to(rank).unsqueeze(1)
            label = label.to(rank)
            
            # sample timesteps

            # sample noisy kps
            map_feat = ddp_net.module.map_encoder(tra_map)
            pred = ddp_net(l, map_feat).squeeze(1)
            
            
            loss = loss_fn(pred, label)
            
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
    parser.add_argument("--master_port", type=str, help="port for master", default='6665')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_train_timesteps", type=int, default=100) # denoising steps
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ema_decay", type=float, default=0.75)
    parser.add_argument("--use_adamw", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
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