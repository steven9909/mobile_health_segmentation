from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import transforms.transform as T
from data.dataset import CuffDataset
from models.unet import UNet
from util.arguments import ArgParser


def show_image(image, mask):
    import matplotlib.pyplot as plt

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image.detach().cpu().permute(1, 2, 0))
    ax2.imshow(mask.detach().cpu().squeeze())
    plt.show()


if __name__ == "__main__":
    args = ArgParser().parse_args()

    PATCH_SIZE = 256

    SAVE_EPOCH = 1000
    CHECK_VAL_EPOCH = 50

    # Using a single worker seems like it has the best performance
    num_workers = 0

    img_dir = Path("original/")
    mask_dir = Path("annotated/")
    img_val_dir = Path("validation/original/")
    mask_val_dir = Path("validation/annotated/")
    BASE_OUTPUT = Path("output")
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = BASE_OUTPUT / "unet.pth"
    TEMP_PATH = BASE_OUTPUT / "temp.pth"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_paths = [file for file in img_dir.iterdir() if not file.name.startswith(".")]
    mask_paths = [file for file in mask_dir.iterdir() if not file.name.startswith(".")]
    img_val_paths = [file for file in img_val_dir.iterdir() if not file.name.startswith(".")]
    mask_val_paths = [file for file in mask_val_dir.iterdir() if not file.name.startswith(".")]

    df = pd.DataFrame({"image_path": img_paths, "mask_paths": mask_paths}, dtype=str)
    df_val = pd.DataFrame("image_path": img_val_paths, "mask_paths": mask_val_paths, dtype=str)

    writer = SummaryWriter()

    transforms_train = T.DualCompose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomShiftScaleRotate(),
            T.RandomColorJitter(),
        ]
    )

    train_dataset = CuffDataset(df, transforms=transforms_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=36,
        shuffle=True,
        num_workers=num_workers,
    )

    validate_dataset = CuffDataset(df_val, transforms=None)
    val_dataloader = DataLoader(
        validate_dataset,
        batch_size=3,
        num_workers=num_workers,
    )
    print(f"Using batch size of 36, and {num_workers} workers")

    model = UNet(3, 1)

    max_epochs = 10000
    init_lr = 1e-3

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
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = sigmoid_focal_loss

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
            
            if epoch % CHECK_VAL_EPOCH == 0:
                model.eval()
                with torch.no_grad():
                    for i, (data, mask) in enumerate(val_dataloader):
                        data = data.to(device)
                        mask = mask.to(device)

                        output = model(data)
                        loss = loss_fn(output, mask)

                        writer.add_scalar(
                            "val_loss",
                            loss.item(),
                            i + (epoch * len(val_dataloader)),
                        )
                        writer.close()
                model.train()
                
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
