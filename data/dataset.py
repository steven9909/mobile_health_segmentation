from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.functional import normalize
from torchvision import transforms


class CuffDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx, 0]).convert("RGB")
        mask = Image.open(self.df.iloc[idx, 1]).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        image = transforms.ToTensor()(np.array(image))
        mask = transforms.ToTensor()(np.array(mask))

        image = normalize(
            image,
            mean=(0.5687, 0.5434, 0.5152),
            std=(0.2508, 0.2399, 0.2307),
        )

        return image, mask.squeeze(0).long()
