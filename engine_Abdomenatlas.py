import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening
import sys
from utils import test_single_volume,test_single_volume_test
import time
import timm
import torch.nn as nn
from tqdm import tqdm

class SwinTransformerForSegmentation(nn.Module):
    def __init__(self, num_classes=26):
        super(SwinTransformerForSegmentation, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, in_chans=1)
        self.backbone.head = nn.Identity()  

        self.backbone_output_channels = 1024  

        self.upsample1 = nn.ConvTranspose2d(self.backbone_output_channels, 512, kernel_size=2, stride=2)  
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   
        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    
        self.final_upsample = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2) 

    def forward(self, x):

        x = self.backbone.forward_features(x)  # (batch, 7, 7, 1024)

        x = x.permute(0, 3, 1, 2)  

        x = self.upsample1(x)  # (batch, 512, 14, 14)
        x = self.upsample2(x)  # (batch, 256, 28, 28)
        x = self.upsample3(x)  # (batch, 128, 56, 56)
        x = self.upsample4(x)  # (batch, 64, 112, 112)
        x = self.final_upsample(x)  # (batch, num_classes, 224, 224)
        
        return x

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
    
    loss_list = []
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100, position=0, leave=True,file=sys.stdout)
    
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)

        train_loader.set_postfix({'Loss': f'{mean_loss:.4f}', 'LR': f'{now_lr:.6f}'})

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            if logger:
                logger.info(log_info)
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    if logger:
        logger.info(log_info)
    return mean_loss






def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            log_info = 'idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
            print(log_info)
            if logger:
                logger.info(log_info)
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            log_info = 'Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1])
            print(log_info)
            if logger:
                logger.info(log_info)
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        if logger:
            logger.info(log_info)
    
    return performance, mean_hd95




def val_one_epoch_test(test_datasets,
                       test_loader,
                       model,
                       epoch,
                       logger,
                       config,
                       test_save_path,
                       val_or_test=False):
    stime = time.time()

    # Ensure config.device is defined
    if not hasattr(config, 'device'):
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    batch_size = 8  # Adjusted to 8 samples per batch

    metric_list = 0.0
    i_batch = 0

    # Initialize CUDA streams
    streams = [torch.cuda.Stream() for _ in range(batch_size)]  # Adjust to 8 streams

    with torch.no_grad():
        batch = []
        for i, data in enumerate(tqdm(test_loader, desc="Loading and Validating")):
            batch.append(data)
            # Process the batch when it's full or at the end of the loader
            if len(batch) == batch_size or i == len(test_loader) - 1:
                # Show progress for validation
                with tqdm(total=len(batch), desc=f"Validating batch {i_batch + 1}") as pbar:
                    for idx, data in enumerate(batch):
                        stream = streams[idx % len(streams)]  # Assign data to a CUDA stream
                        with torch.cuda.stream(stream):  # Perform operations in the stream
                            img = data['image'].to(config.device, non_blocking=True)
                            msk = data['label'].to(config.device, non_blocking=True)
                            case_name = data['case_name'][0]

                            # Use mixed precision for faster computation
                            with torch.cuda.amp.autocast():
                                metric_i = test_single_volume(
                                    img,
                                    msk,
                                    model,
                                    classes=config.num_classes,
                                    patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path,
                                    case=case_name,
                                    z_spacing=config.z_spacing,
                                    val_or_test=val_or_test
                                )

                            # Accumulate metrics
                            metric_list += np.array(metric_i)

                            # Log metrics for this case
                            log_info = 'idx %d case %s mean_dice %f mean_hd95 %f' % (
                                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]
                            )
                            print(log_info)
                            if logger:
                                logger.info(log_info)
                            i_batch += 1

                            # Update progress bar
                            pbar.update(1)

                    # Synchronize all streams to ensure completion
                    for stream in streams:
                        stream.synchronize()

                # Clear the batch for the next iteration
                batch = []

        # Compute overall metrics
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]

        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        if logger:
            logger.info(log_info)

    return performance, mean_hd95
