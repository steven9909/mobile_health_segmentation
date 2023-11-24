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

# example 
def show_image(image, mask):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image.detach().cpu().permute(1, 2, 0))
    plt.imshow(mask.detach().cpu().squeeze(), alpha=0.5)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    args = ArgParser().parse_args()

    PATCH_SIZE = 512

    SAVE_EPOCH = 500
    CHECK_VAL_EPOCH = 50
    PRINT_EPOCH = 10
    BATCH_SIZE = 32

    MAX_EPOCHS = 40000
    init_lr = 5e-6

    # Using a single worker seems like it has the best performance
    num_workers = 0

    img_dir = Path("./original")
    mask_dir = Path("./annotated")
    img_val_dir = Path("./validation/original")
    mask_val_dir = Path("./validation/annotated")
    BASE_OUTPUT = Path("./output")
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = BASE_OUTPUT / "unet.pth"
    TEMP_PATH = BASE_OUTPUT / "temp.pth"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_paths = [file for file in img_dir.iterdir() if not file.name.startswith(".")]
    mask_paths = [file for file in mask_dir.iterdir() if not file.name.startswith(".")]
    img_val_paths = [
        file for file in img_val_dir.iterdir() if not file.name.startswith(".")
    ]
    mask_val_paths = [
        file for file in mask_val_dir.iterdir() if not file.name.startswith(".")
    ]

    df = pd.DataFrame({"image_path": img_paths, "mask_paths": mask_paths}, dtype=str)
    df_val = pd.DataFrame(
        {"image_path": img_val_paths, "mask_paths": mask_val_paths}, dtype=str
    )

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
    )

    validate_dataset = CuffDataset(df_val, transforms=None)
    val_dataloader = DataLoader(
        validate_dataset,
        batch_size=1,
        num_workers=num_workers,
    )
    print(f"Using batch size of {BATCH_SIZE}, and {num_workers} workers")

    # model = UNet(3, 1)
    model = torch.hub.load(
        "milesial/Pytorch-UNet",
        "unet_carvana",
        pretrained=True,
        scale=1,
    )

    train_dataloader_len = len(train_dataloader)

    optim = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=5e-4)
    scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / MAX_EPOCHS)
    start_epoch = 0

    if args.continue_training == "y":
        print(f"Continuing training from checkpoint defined at: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH)["model"])
        scheduler.load_state_dict(torch.load(MODEL_PATH)["scheduler"])
        start_epoch = torch.load(MODEL_PATH)["epoch"]

    model = model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # loss_fn = sigmoid_focal_loss

    def dice_loss(input, target, classes):
        smooth = 1.0

        input = torch.softmax(input, 1)
        dice = 0
        for class_i in range(classes):
            input_f = input[:, class_i, :, :]
            target_f = target == class_i

            intersect = (input_f * target_f).sum()
            union = input_f.sum() + target_f.sum() + smooth

            dice += (2.0 * intersect + smooth) / union

        return 1 - dice / classes

    for epoch in tqdm(range(start_epoch, MAX_EPOCHS)):
        for i, (data, mask) in enumerate(train_dataloader):
            optim.zero_grad()

            data = data.to(device)
            mask = mask.to(device)

            output = model(data)
            loss = dice_loss(output, mask, 2)
            # show_image(data[0], mask[0])

            loss.backward()
            optim.step()

            if i + (epoch * train_dataloader_len) % PRINT_EPOCH == 0:
                writer.add_scalar(
                    "loss",
                    loss.item(),
                    i + (epoch * train_dataloader_len),
                )

        scheduler.step()
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

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
                    loss = dice_loss(output, mask, 2)

                    writer.add_scalar(
                        "val_loss",
                        loss.item(),
                        i + (epoch * len(val_dataloader)),
                    )
            model.train()

        writer.close()

    torch.save(
        {
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        MODEL_PATH,
    )
