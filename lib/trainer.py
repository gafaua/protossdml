from collections import deque
from os import makedirs
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lib.losses import infoNCELoss, kde_regularization_vmf
from lib.siamese_ema import SiameseEMA
from lib.vision_transformer import vit_small


class Trainer:
    """Trainer class for ProtoSSDML.
    """
    def __init__(self, loader, args) -> None:
        self.model = SiameseEMA(
            base_encoder=vit_small(),
            proj_dim=args.proj_dim,
            out_dim=args.out_dim,
            momentum=args.momentum,
            projection_head=args.projection_head,
        )

        if args.multigpu:
            self.model = nn.DataParallel(self.model)

        self.model.to(args.device)
        self.model.train()

        #### MOCO PARAMETERS ####
        self.queue_size = args.queue_size
        self.queue = torch.randn(args.out_dim, self.queue_size, device=args.device)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.T = args.temperature

        self.use_kde_vmf_regularizer = args.use_kde_vmf_regularizer

        self.device = args.device
        self.train_loader = loader

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs

        self.save_interval = args.save_interval
        self.save_dir = join(args.save_dir, args.run_name)

        makedirs(self.save_dir, exist_ok=True)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs)

        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)
        else:
            self.model_params = list(self.model.parameters())
            self.name = args.run_name
            self.step = 0
            self.epoch = 0

        num_parameters = sum(p.numel() for p in self.model_params)

        print("#"*5+f"Initialized training for model with {num_parameters:,} parameters" + "#"*5)


    def train(self):
        train_epochs = range(self.epoch, self.num_epochs)
        for _ in train_epochs:
            self._run_train_epoch()
            self.epoch += 1
            if self.epoch % self.save_interval == 0:
                self._save_checkpoint()
            else:
                self._save_checkpoint(name="last")

        self._save_checkpoint()


    def _train_batch_loss(self, batch, train=True):
        x1, x2 = batch
        q, k = self.model(x1.to(self.device), x2.to(self.device))

        queue = self.queue if train else self.queue_val
        queue_ptr = self.queue_ptr if train else self.queue_ptr_val

        loss = infoNCELoss(q, k, queue, self.T)
        losses = dict(loss=loss)

        if self.use_kde_vmf_regularizer:
            reg_kde = kde_regularization_vmf(q)
            loss += 0.1 * reg_kde
            losses["reg_kde"] = reg_kde.item()

        self._dequeue_and_enqueue(k, queue, queue_ptr)

        return loss, losses


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        queue_ptr[0] = ptr


    def _run_train_epoch(self):
        self.model.train()

        losses = dict(loss=deque(maxlen=100))
        avg_losses = dict()
        pbar = tqdm(self.train_loader, desc=f"Train Epoch: {self.epoch+1} ", ncols=120)

        for batch in pbar:
            self.opt.zero_grad()

            loss, info_dict = self._train_batch_loss(batch)

            loss.backward()
            self.opt.step()

            info_dict["loss"] = loss.item()
            info_dict["LR"] = self.opt.param_groups[0]["lr"]

            for l_name in info_dict:
                if l_name not in losses:
                    losses[l_name] = deque(maxlen=100)
                losses[l_name].append(info_dict[l_name])
                avg_losses[l_name] = np.mean(losses[l_name])

            pbar.set_postfix(dict(avg_loss=avg_losses["loss"], **info_dict))
            self.step += 1

        self.lr_scheduler.step()


    def _save_checkpoint(self, name=None):
        model_dict = self.model.state_dict()

        data = dict(
            model_dict=model_dict,
            opt_dict=self.opt.state_dict(),
            lr_scheduler_dict=self.lr_scheduler.state_dict(),
            epoch=self.epoch,
            step=self.step,
            name=self.name,
        )

        if name is None:
            save_path =  join(self.save_dir, f"checkpoint_{self.step}.pth")
        else:
            save_path = join(self.save_dir, f"checkpoint_{name}.pth")

        torch.save(data, save_path)
        print(f"Checkpoint saved in {save_path}")


    def _load_checkpoint(self, path):
        data = torch.load(path)
        self.model.load_state_dict(data["model_dict"])
        self.model_params = list(self.model.parameters())
        self.opt.load_state_dict(data["opt_dict"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler_dict"])
        self.epoch = data["epoch"]
        self.step = data["step"]
        self.name = f"{data['name']}_resumed"

        print("="*100)
        print(f"Resuming training from checkpoint {path}")
        print("="*100)
