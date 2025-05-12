import argparse
import configparser
from datetime import datetime
from pprint import pprint

import numpy as np
import torch


def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_file", type=str, help="Config file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multigpu", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))

    # Dataloader
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--split_file", type=str, help="Path to JSON file with WSI train/test split. File should contain a dict in the form {str: List[Path]}")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wsi_resolution", type=str, help="Magnification objective resolution(s) to use to extract patches. Supports multi-resolution training with comma separated values")

    # Model
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to load")

    # Moco
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--queue_size", type=int, default=65536)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--temperature", type=float, default=0.07)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_interval", type=int, default=10)

    # Proposed modification
    parser.add_argument("--fixed_scale", default=1, type=int)             # 1: fixed scale crop, 0: random resized crop (classic)
    parser.add_argument("--random_sharpen", default=1, type=int)          # 1: use random high pass filter in augmentations, 0: don't
    parser.add_argument("--projection_head", default=0, type=int)         # 1: add a projection in the siamese setting, 0: don't
    parser.add_argument("--use_kde_vmf_regularizer", default=1, type=int) # 1: use kde von Moses Ficher regularizer

    ## LR Schedulers
    parser.add_argument("--max_lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--steps", type=str)

    args = parser.parse_args()

    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        defaults = {}
        sections = ["Dataset", "Model", "Training"]
        for s in sections:
            if config.has_section(s):
                defaults.update(dict(config.items(s)))
        parser.set_defaults(**defaults)
        args = parser.parse_args() # Overwrite arguments

    # Pre-Process some args:
    if args.wsi_resolution is not None:
        args.wsi_resolution = [int(res) for res in args.wsi_resolution.split(",")]

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available")
            args.device = "cpu"

    pprint(args.__dict__)

    print("#"*10 + f"Device: {args.device}" + "#"*10)
    print("#"*10 + f"Available GPUs: {torch.cuda.device_count()}" + "#"*10)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Seed set: {args.seed}")

    return args
