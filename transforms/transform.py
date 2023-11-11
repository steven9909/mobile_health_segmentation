import torch
from torchvision.transforms import functional as F


class DualCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip(object):
    def __call__(self, image, mask):
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip(object):
    def __call__(self, image, mask):
        if torch.rand(1) < 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class RandomShiftScaleRotate(object):
    def __call__(self, image, mask):
        if torch.rand(1) < 0.5:
            angle = torch.randint(-45, 45, (1,)).item()
            translate = torch.rand(1).item() * 0.1
            scale = torch.rand(1).item() * 0.4 + 0.4
            shear = torch.randn(1).item() * 10 - 5
            image = F.affine(
                image, angle, (translate, translate), scale, (shear, shear)
            )
            mask = F.affine(mask, angle, (translate, translate), scale, (shear, shear))
        return image, mask


class RandomColorJitter(object):
    def __call__(self, image, mask):
        if torch.rand(1) < 0.5:
            brightness = torch.rand(1).item() * 0.1 + 0.95
            contrast = torch.rand(1).item() * 0.1 + 0.95
            saturation = torch.rand(1).item() * 0.1 + 0.95
            hue = torch.rand(1).item() * 0.2 - 0.1
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)
            image = F.adjust_hue(image, hue)
        return image, mask
