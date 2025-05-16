import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.dataset import RandomGenerator
from engine_Abdomenatlas import *

from models.vmunet.vmunet import VMUNet

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

from utils import *
from configs.config_setting_Abdomenatlas import setting_config
import timm
import warnings
warnings.filterwarnings("ignore")

def main(config,best_model_path):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    outputs = os.path.join(config.work_dir, 'outputs')

    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('validate', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    torch.cuda.empty_cache()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('#----------Preparing dataset----------#')
    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    print('#----------Loading Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else:
        raise ValueError('Please provide a correct network.')
    
    print('#----------Loading checkpoint----------#')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict, strict=False)  
        model = model.to(device)
        
        logger.info(f"Loaded model weights from {best_model_path}")
    else:
        raise FileNotFoundError(f"Checkpoint {best_model_path} not found!")


    print('#----------Validating----------#')
    mean_dice, mean_hd95 = val_one_epoch(
        val_dataset,
        val_loader,
        model,
        epoch=0,  
        logger=logger,
        config=config,
        test_save_path=outputs,
        val_or_test=True
    )

    logger.info(f"Validation Complete. Mean Dice: {mean_dice:.4f}, Mean HD95: {mean_hd95:.4f}")

if __name__ == '__main__':
    config = setting_config
    config.distributed = False
    best_model_path = "./VM_UNet/VM-UNet-main/results/vmunet_AbdomenAtlas_Friday_16_May_2025_14h_11m_57s/checkpoints/best.pth"  
    main(config,best_model_path)
