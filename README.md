## This is my assignment in deep learning course about semantic segmantation
Nguyen Viet Minh 20214917

# Infer
Step 1: Join competition https://www.kaggle.com/competitions/bkai-igh-neopolyp to create notebook

Step 2: Download model and libraries

```python
!pip install torchgeometry
!pip install torchsummary
!pip install segmentation-models-pytorch
import requests
import os

url = 'https://drive.google.com/uc?id=1qhbGzKUULYFMWl3I2R459Nk9VKyLJ9rS&export=download&confirm=t&uuid=12eaf101-0796-4f5b-813b-cbe20b5dbde0'

save_dir = '../working/'

response = requests.get(url)

with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)

```
Step 3: Infer
```python
!git clone https://github.com/Min-KiD/SSdeepLHW
!cp ../working/model.pth ../working/SSdeepLHW
!python ../working/SSdeepLHW/infer.py
```
