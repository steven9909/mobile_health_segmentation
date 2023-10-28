from model import UNet
from dataset import CuffDataset

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import glob as gl
import os
import os.path

PATCH_SIZE = 256

img_dir = "original/"
mask_dir = "annotated/"
BASE_OUTPUT="output"
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth")

img_paths = [file for file in Path(img_dir).iterdir() if not file.name.startswith(".")]
mask_paths = [
    file for file in Path(mask_dir).iterdir() if not file.name.startswith(".")
]

df = pd.DataFrame({"image_path": img_paths, "mask_paths": mask_paths}, dtype=str)

transforms_train = A.Compose(
    [
        A.RandomCrop(width=PATCH_SIZE, height=PATCH_SIZE, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        ToTensorV2(),
    ]
)

transforms_val = A.Compose([ToTensorV2(),])

# Split df into train and test data
train_df, val_df = train_test_split(df, test_size=0.2)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = CuffDataset(train_df, transforms=transforms_train)
train_dataloader = DataLoader(train_dataset, batch_size=28, shuffle=False)

val_dataset = CuffDataset(val_df, transforms=transforms_val)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

if not os.path.exists(MODEL_PATH):
    model = UNet(3, 1)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()

    print('Training')
    max_epochs = 4

    for epoch in tqdm(range(max_epochs)):
        for data, mask in train_dataloader:
            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output, mask)
            loss.backward()
            optim.step()

    #torch.save(model, MODEL_PATH)
else:
    model = torch.load(MODEL_PATH)

print('Testing')
model.eval()
with torch.no_grad():
    for data, mask in val_dataloader:
        outputs = model(data)
        print(loss_fn(outputs, mask))

'''
def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())

def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc/batch_size
'''
