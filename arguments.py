import argparse


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training script for UNet")

        self.parser.add_argument(
            "--continue_training",
            type=str,
            default="n",
            help="Whether to continue training from checkpoint",
        )

    def parse_args(self):
        return self.parser.parse_args()
