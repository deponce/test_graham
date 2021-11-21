import torchvision
from torchvision import models, datasets, transforms
from utils import load_data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print('start load')
data_path = '~/scratch/deponce/data/imagenet12'
train_data, test_data, val_data = load_data(data_path)

import numpy as np
arr = np.array([x for x in range(10)])
np.save('./test_arr',arr)