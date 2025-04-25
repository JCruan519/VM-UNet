# VM-UNet
This is the official code repository for "VM-UNet: Vision Mamba UNet for Medical
Image Segmentation". {[Arxiv Paper](https://arxiv.org/abs/2402.02491)}

## Abstract
In the realm of medical image segmentation, both CNN-based and Transformer-based models have been extensively explored. However, CNNs exhibit limitations in long-range modeling capabilities, whereas Transformers are hampered by their quadratic computational complexity. Recently, State Space Models (SSMs), exemplified by Mamba, have emerged as a promising approach. They not only excel in modeling long-range interactions but also maintain a linear computational complexity. In this paper, leveraging state space models, we propose a U-shape architecture model for medical image segmentation, named Vision Mamba UNet (VM-UNet). Specifically, the Visual State Space (VSS) block is introduced as the foundation block to capture extensive contextual information, and an asymmetrical encoder-decoder structure is constructed. We conduct comprehensive experiments on the ISIC17, ISIC18, Synapse and Abdomenatlas datasets, and the results indicate that VM-UNet performs competitively in medical image segmentation tasks. To our best knowledge, this is the first medical image segmentation model constructed based on the pure SSM-based model. We aim to establish a baseline and provide valuable insights for the future development of more efficient and effective SSM-based segmentation systems.

## 0. Main Environments
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The fixed version of the .whl files of causal_conv1d and mamba_ssm could be found here. {[Google Drive](https://drive.google.com/drive/folders/1fW8KcW29tIDQ7yL_2GNbl5bzU8dboold?usp=drive_link)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
    
  - test_vol_h5
    - casexxxx.npy.h5
    
  - train_npz
    - casexxxx_slicexxx.npz
    
      

### Abdomenatlas datasets

- For the Abdomenatlas datasets, we organize the data format as the synapse datasets.
- After downloading the datasets, you are supposed to preprocess the data into the following format.
- './data/Abdomanatlas/'
  - lists
    - list_abdomenatlas
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 3. Training the VM-UNet Model
- To train the VM-UNet model, navigate to the VM-UNet directory and execute the appropriate training script based on your dataset and hardware configuration.

### Single GPU Training

- ISIC17 or ISIC18 Datasets:
```bash
python train.py  
```
This command trains and tests the VM-UNet model on the specified dataset.

- Synapse Dataset:
```bash
python train_synapse.py
```
This command trains and tests the VM-UNet model on the Synapse dataset.

### Multi-GPU Training

- Specify GPUs:
In the train_Abdomenatlas.py script, set the CUDA_VISIBLE_DEVICES environment variable to select the GPUs you wish to use. For example:
 ```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
```
This configuration uses GPUs 4, 5, 6, and 7.

- Start Training:
 ```bash
python train_Abdomenatlas.py
```
### Configuration Note
For modifications to the training setup or to adapt the script for other datasets, refer to and adjust the settings in ./configs/config_setting_Abdomenatlas.py. or other configs.

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. About the validation
- You can simply run the 'validation.py' to get the output and dice scores and HD95.
- However, to reduce computation time in multi-class segmentation tasks, you can comment out the HD95 calculation in the calculate_metric_percase function inside the utils module (specifically the part where hd95 is computed) and temporarily set hd95 = 0.

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.
