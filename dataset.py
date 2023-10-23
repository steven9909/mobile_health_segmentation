from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.functional import normalize


class CuffDataset(Dataset):
    def __init__(self, df, transforms):
        # df contains the paths to all files
        self.df = df
        # transforms is the set of data augmentation operations
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.df.iloc[idx, 0]), dtype=np.float32) / 255
        mask = np.array(Image.open(self.df.iloc[idx, 1]), dtype=np.float32) / 255

        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]  # Dimension (3, 255, 255)
        mask = augmented["mask"]  # Dimension (255, 255)
        image = normalize(
            image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True
        )
        mask = np.expand_dims(mask, axis=0)
        return image, mask
