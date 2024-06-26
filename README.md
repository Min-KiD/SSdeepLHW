## This is my assignment in deep learning course about semantic segmantation
Nguyen Viet Minh 20214917

Short description: Built and trained a deep learning model for semantic segmentation problem in Medicine. It detect neoplasm part and classify which kind it belong to

Detail description: Turn all image to fixed size Tensor then data augmentation using albumentations, image and mask are separated and normalized. I choose model Unet++ using segmentation_models_pytorch library combine with DataParallel and Focal Loss for training process, optimizer using AdamW and scheduler using StepLR. The Checkpoint is created to save model parameter after each time test batch loss is lower than the previous one 

# Infer
Step 1: Join competition https://www.kaggle.com/competitions/bkai-igh-neopolyp to create notebook

Step 2: Download model and libraries

```python
!pip install torchgeometry
!pip install segmentation-models-pytorch
import requests
import os

url = 'https://drive.google.com/uc?id=1qhbGzKUULYFMWl3I2R459Nk9VKyLJ9rS&export=download&confirm=t&uuid=12eaf101-0796-4f5b-813b-cbe20b5dbde0'

save_dir = '/kaggle/working/'

response = requests.get(url)

with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)

```
Step 3: Infer
```python
!git clone https://github.com/Min-KiD/SSdeepLHW
!cp /kaggle/working/model.pth /kaggle/working/SSdeepLHW
!python /kaggle/working/SSdeepLHW/infer.py
```
