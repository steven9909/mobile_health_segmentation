from model import UNet
from dataset import CuffDataset
from arguments import ArgParser

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    args = ArgParser().parse_args()

    PATCH_SIZE = 256

    SAVE_EPOCH = 500

    # Using a single worker seems like it has the best performance
    num_workers = 0

    img_dir = Path("original/")
    mask_dir = Path("annotated/")
    BASE_OUTPUT = Path("output")
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = BASE_OUTPUT / "unet.pth"
    TEMP_PATH = BASE_OUTPUT / "temp.pth"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_paths = [file for file in img_dir.iterdir() if not file.name.startswith(".")]
    mask_paths = [file for file in mask_dir.iterdir() if not file.name.startswith(".")]

    df = pd.DataFrame({"image_path": img_paths, "mask_paths": mask_paths}, dtype=str)

    writer = SummaryWriter()

    transforms_train = A.Compose(
        [
            A.Resize(
                width=PATCH_SIZE, height=PATCH_SIZE, interpolation=cv2.INTER_NEAREST
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25
            ),
            ToTensorV2(),
        ]
    )

    transforms_val = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    train_dataset = CuffDataset(df, transforms=transforms_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=36,
        shuffle=True,
        num_workers=num_workers,
    )
    print(f"Using batch size of 36, and {num_workers} workers")

    model = UNet(3, 1)

    max_epochs = 3000
    init_lr = 9e-4

    train_dataloader_len = len(train_dataloader)

    optim = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_epochs)
    start_epoch = 0

    if args.continue_training == "y":
        print(f"Continuing training from checkpoint defined at: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH)["model"])
        scheduler.load_state_dict(torch.load(MODEL_PATH)["scheduler"])
        start_epoch = torch.load(MODEL_PATH)["epoch"]

    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    try:
        for epoch in tqdm(range(start_epoch, max_epochs)):
            for i, (data, mask) in enumerate(train_dataloader):
                optim.zero_grad()

                data = data.to(device)
                mask = mask.to(device)

                output = model(data)
                loss = loss_fn(output, mask)

                loss.backward()
                optim.step()

                writer.add_scalar(
                    "loss",
                    loss.item(),
                    i + (epoch * train_dataloader_len),
                )
            scheduler.step()
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
            writer.close()

            if epoch % SAVE_EPOCH == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    BASE_OUTPUT / f"temp_{epoch}",
                )
        torch.save(
            {
                "model": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            MODEL_PATH,
        )
    except KeyboardInterrupt:
        optim.zero_grad()
        torch.save(
            {
                "model": model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            TEMP_PATH,
        )
