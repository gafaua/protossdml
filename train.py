import json
from os.path import join

from lib.augmentations import get_augmentations
from lib.dataloader import make_bunch_wsi_loader
from lib.parser import parse_train_args
from lib.trainer import Trainer


def train(args):
    with open(args.split_file, "r") as f:
        splits = json.load(f)

    train_wsis = [join(args.data_dir, wsi) for wsi in splits["train"]]

    train_dataloader = make_bunch_wsi_loader(wsi_paths=train_wsis,
                                             wsi_resolution=args.wsi_resolution,
                                             patch_size=args.patch_size,
                                             transforms=get_augmentations(args),
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True,
                                             overlap=0.0)

    Trainer(train_dataloader, args).train()

if __name__ == "__main__":
    train(parse_train_args())
