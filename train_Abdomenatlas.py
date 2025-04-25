import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torch.multiprocessing as mp

from datasets.dataset import RandomGenerator
from engine_synapse import *

from models.vmunet.vmunet import VMUNet

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
from utils import *
from configs.config_setting_Abdomenatlas import setting_config
import timm
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

def main_worker(local_rank, config):
    print('#----------GPU init----------#')
    config.local_rank = local_rank

    dist.init_process_group(
        backend='nccl',  
        init_method='tcp://127.0.0.1:23457',
        rank=config.local_rank,
        world_size=torch.cuda.device_count()
    )

    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    sys.path.append(config.work_dir + '/')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')

    if rank == 0:
        print('#----------Creating logger----------#')
        log_dir = os.path.join(config.work_dir, 'log')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        global logger
        logger = get_logger('train', log_dir)

        log_config_info(config, logger)

    print('#----------Preparing dataset----------#')
    train_dataset = config.datasets(
        base_dir=config.data_path, 
        list_dir=config.list_dir, 
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]
        )
    )
    train_sampler = DistributedSampler(train_dataset) if config.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // torch.cuda.device_count() if config.distributed else config.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=config.num_workers,
        sampler=train_sampler
    )


    print('#----------Preparing Models----------#')
    model = SwinTransformerForSegmentation(num_classes=26)
    
    model = model.to(device_id)
    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device_id], output_device=device_id, broadcast_buffers=False, find_unused_parameters=False)
    else:
        model = model.cuda()

    print('#----------Preparing loss, optimizer, scheduler, and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = float('inf')
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(os.path.join(checkpoint_dir, 'latest.pth')) and rank == 0:
        print('#----------Resuming Model and Other params----------#')
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'latest.pth'), map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model_state_dict']

        if config.distributed:
            model.module.load_state_dict(model_state_dict, strict=False)
        else:
            model.load_state_dict(model_state_dict, strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'Resuming model from {os.path.join(checkpoint_dir, "latest.pth")}. Resume_epoch: {saved_epoch}, Min_loss: {min_loss:.4f}, Min_epoch: {min_epoch}, Loss: {loss:.4f}'
        logger.info(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        if config.distributed:
            train_sampler.set_epoch(epoch)

        loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger if rank == 0 else None,
            config,
            scaler=scaler
        )


        if rank == 0 and loss < min_loss:
            min_loss = loss
            min_epoch = epoch
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict() if config.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'best.pth')
            )

        if rank == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict() if config.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')
            )

if __name__ == '__main__':
    config = setting_config
    config.distributed = True 
    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(config,))