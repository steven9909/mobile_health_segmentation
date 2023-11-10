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

        mask = np.expand_dims(mask, axis=-1)

        image = transforms.ToTensor()(np.array(image))
        mask = transforms.ToTensor()(np.array(mask))

        image = normalize(
            image,
            mean=(0.5612, 0.5397, 0.5159),
            std=(0.2515, 0.2405, 0.2317),
        )

        return image, mask
