from os import makedirs

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from torchvision.transforms.functional import to_tensor

from lib.vision_transformer import VisionTransformer


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    if type(imgs) == torch.Tensor:
        imgs = imgs.permute(0, 2,3,1).numpy()
        imgs = (imgs * 255).astype(np.uint8)
        imgs = [Image.fromarray(img) for img in imgs]

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def visualize_attention(model: VisionTransformer,
                        img: Image,
                        transforms: nn.Sequential,
                        save_dir: str):
    patch_size = model.patch_embed.patch_size

    img_tensor = to_tensor(img)

    w_featmap = img_tensor.shape[-2] // patch_size
    h_featmap = img_tensor.shape[-1] // patch_size

    inputs = transforms(img_tensor).to(model.cls_token.device).unsqueeze(0)
    attentions = model.get_last_selfattention(inputs)
    nh = attentions.shape[1] #Number of heads
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0),
                                           scale_factor=patch_size,
                                           mode="nearest")[0].cpu().numpy()

    makedirs(save_dir, exist_ok=True)
    images = []
    for i in range(nh):
        mn, mx = attentions[i].min(), attentions[i].max()

        normalized = (attentions[i] - mn) / (mx-mn)
        attn = img * np.repeat(normalized[:, :, np.newaxis], 3, axis=2)
        attn = Image.fromarray((attn).astype(np.uint8)).convert("RGB")
        attn = Image.blend(img, attn, 0.7)
        attn = ImageOps.expand(attn, 1, "black")

        images.append(attn)

    img = make_image_grid(images, int(len(images)/3), 3)
    img.save("attn_viz.png")


def plot_mean_curve(ax, aucs, xs, ys, name, **kwargs):
    mean_x = np.linspace(0, 1, 100)
    ys_interp = []

    for i in range(len(aucs)):
        interp_y = np.interp(mean_x, xs[i], ys[i])
        interp_y[0] = 0.0
        ys_interp.append(interp_y)

    mean_y = np.mean(ys_interp, axis=0)
    mean_y[-1] = 1.0
    mean_auc = auc(mean_x, mean_y)
    std_auc = np.std(aucs)

    ax.plot(
            mean_x,
            mean_y,
            label=r"%s (AUC = %0.3f $\pm$ %0.3f)" % (name, mean_auc, std_auc),
            lw=2,
            alpha=1,
            **kwargs
        )
    std_tpr = np.std(ys_interp, axis=0)
    tprs_upper = np.minimum(mean_y + std_tpr, 1)
    tprs_lower = np.maximum(mean_y - std_tpr, 0)
    ax.fill_between(
            mean_x,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
        )

    ax.grid(True)

def plot_mean_pr_curve(ax, aucs, y_trues, y_scores, name, **kwargs):
    for i in range(len(aucs)):
        precision, recall, _ = precision_recall_curve(y_trues[i], y_scores[i])
        ax.plot(recall,
                precision,
                lw=1,
                alpha=0.3,
                **kwargs)

    all_trues = np.concatenate(y_trues)
    all_scores = np.concatenate(y_scores)
    all_aps = []
    recalls = []
    precisions = []

    for i in range(len(aucs)):
        ap = average_precision_score(y_trues[i], y_scores[i])
        p, r, _ = precision_recall_curve(y_trues[i], y_scores[i])
        all_aps.append(ap)
        recalls.append(r)
        precisions.append(p)

    precision, recall, _ = precision_recall_curve(all_trues, all_scores)

    ax.plot(
            recall,
            precision,
            label=r"%s (AP = %0.3f $\pm$ %0.3f" % (name, np.mean(all_aps), np.std(all_aps)),
            lw=2,
            alpha=1,
            **kwargs
        )

    ax.grid(True)
