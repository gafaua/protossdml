import csv
import json
from glob import glob
from os import makedirs, path
from pprint import pprint
from typing import List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.visualizations import plot_mean_curve, plot_mean_pr_curve
from lib.wsi_analysis.wsi_dataset import WSIDataset


@torch.no_grad()
def _get_embeddings(model: nn.Module,
                    wsi_path: str,
                    patch_size: int,
                    resolution: int,
                    annotations_paths: List[str],
                    annotations_type: str,
                    patho_only:bool,
                    transform):

    wsi_dataset = WSIDataset(wsi_path,
                            patch_size,
                            resolution,
                            transforms=transform,
                            annotations_files=annotations_paths,
                            annotations_type=annotations_type,
                            patho_only=patho_only,
                            verbose=False)

    loader = DataLoader(wsi_dataset,
                        batch_size=128,
                        shuffle=False,
                        drop_last=False,
                        num_workers=8)
    embeddings = [[], [], []]

    for batch in loader:
        img, anomalies, coords = batch
        embs = model(img.to("cuda"))
        embs = embs.cpu().numpy()
        embs = np.squeeze(embs)

        if len(anomalies) == 1:
            embs = [embs]

        coords = list(zip(*coords))
        for i in range(len(anomalies)):
            embeddings[0].append((coords[i][0].item(), coords[i][1].item()))
            embeddings[1].append(anomalies[i].item())
            embeddings[2].append(embs[i])

    coords, anomalies, embeddings = embeddings

    return coords, anomalies, embeddings


def generate_support_set(
                     model,
                     transform,
                     slides_for_proto: List[str],
                     base_annotations_path,
                     export_name,
                     patch_size=224,
                     resolution=20,
                     mean_per_wsi=False,
                     embeddings_cache=None) -> None:
    model = model.to("cuda")
    model.eval()
    embeddings = []
    number_of_support_patches = 0

    for wsi in slides_for_proto:
        name = wsi.split("/")[-1].split(".")[0]
        if embeddings_cache is not None and name in embeddings_cache:
            embeddings.extend(embeddings_cache[name])
            print("Used cached embeddings")
            continue

        annotations = glob(f"{base_annotations_path}/{name}.xml")
        _, _, embs = _get_embeddings(model=model,
                                     wsi_path=wsi,
                                     patch_size=patch_size,
                                     resolution=resolution,
                                     annotations_paths=annotations,
                                     annotations_type="camelyon_xml",
                                     patho_only=True,
                                     transform=transform)
        if len(embs) == 0:
            continue

        number_of_support_patches += len(embs)

        if mean_per_wsi:
            mean_vec = np.array(embs).mean(axis=0)
            embeddings.extend([mean_vec])
            if embeddings_cache is not None:
                embeddings_cache[name] = [mean_vec]
        else:
            embeddings.extend(embs)
            if embeddings_cache is not None:
                embeddings_cache[name] = embs

    embeddings = np.array(embeddings)
    np.savez(export_name, embeddings)
    return number_of_support_patches


def project_evaluation_slides(model,
                              base_embeddings_path,
                              annotations_path,
                              evaluation_slides,
                              patch_size,
                              resolution,
                              transform):
    print("PROJECTING EVALUATION SLIDES PATCHES TO REPRESENTATION SPACE")
    model.to("cuda")
    makedirs(base_embeddings_path, exist_ok=True)

    for label, slides in evaluation_slides.items():
        pb = tqdm(slides, desc=label)
        for slide in pb:
            name = slide.split("/")[-1].split(".")[0]
            pb.set_postfix_str(f"|{name}")

            embeddings_path=f"{base_embeddings_path}/{name}"
            if path.exists(f"{embeddings_path}.npz"):
                continue

            annotations = glob(path.join(annotations_path,f"{name}.xml"))

            coords, anomalies, embeddings = _get_embeddings(model=model,
                                                            wsi_path=slide,
                                                            patch_size=patch_size,
                                                            resolution=resolution,
                                                            annotations_paths=annotations if len(annotations) > 0 else None,
                                                            annotations_type="camelyon_xml",
                                                            patho_only=False,
                                                            transform=transform)

            np.savez(embeddings_path, coords, anomalies, embeddings)

def get_prototypes(model, transform, slides_for_proto, cv_size, patch_size, base_embeddings_path):
    print("PROJECTING SUPPORT SET TO REPRESENTATION SPACE TO OBTAIN PROTOTYPES")

    prototypes = []
    sizes = []

    slides_for_proto = np.array_split(slides_for_proto, cv_size)
    for i, slides in enumerate(tqdm(slides_for_proto, desc="Projecting support sets to representation space")):
        embeddings_name = f"{base_embeddings_path}/embeddings_tumor_{i}"

        size = generate_support_set(model,
                                    transform,
                                    slides,
                                    base_annotations_path="/store2/travail/data/CAMELYON16/annotations/",
                                    export_name=embeddings_name,
                                    mean_per_wsi=True,
                                    patch_size=patch_size)

        support = np.load(f"{embeddings_name}.npz", allow_pickle=True)["arr_0"]
        sizes.append(size)
        prototypes.append(np.mean(support, axis=0))

    print("Support set sizes: ", sizes)
    return prototypes


def cosine_similarity(embedding, prototype):
    return 1+np.dot(prototype, embedding)/(np.linalg.norm(prototype)*np.linalg.norm(embedding))


def get_roc_info(normal_scores, anomaly_scores):
    y_true = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    y_scores = np.array(normal_scores + anomaly_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr), fpr, tpr


def get_evaluation_data(normal_scores, anomaly_scores):
    """
    Returns [ROC AUC, PR AUC, fpr, tpr, precisions, recalls]
    """
    print(f"{len(normal_scores)} normal patches vs {len(anomaly_scores)} patho patches")
    y_true = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    y_scores = np.array(normal_scores + anomaly_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr), auc(recall, precision), fpr, tpr, y_true, y_scores


def evaluate_patch_level_camelyon_cv(model,
                                     model_name,
                                     transform,
                                     annotations_path,
                                     evaluation_slides,
                                     slides_for_proto,
                                     cv_size,
                                     patch_size,
                                     resolution=20,
                                    ):
    base_embeddings_path = f"./evaluation_results/{model_name}"
    makedirs(base_embeddings_path, exist_ok=True)

    project_evaluation_slides(model,
                              base_embeddings_path,
                              annotations_path,
                              evaluation_slides,
                              patch_size,
                              resolution,
                              transform)

    prototypes = get_prototypes(model,
                                transform,
                                slides_for_proto,
                                cv_size, patch_size,
                                base_embeddings_path)

    print("COMPUTING SCORES")
    all_scores = [{1:[], -1:[]} for _ in range(len(prototypes))]
    for i in range(len(prototypes)):
        eval_dir_fold = f"{base_embeddings_path}/init/{i}"
        makedirs(eval_dir_fold, exist_ok=True)

    for label, slides in evaluation_slides.items():
        pb = tqdm(slides, desc=label)
        for slide in pb:
            name = slide.split("/")[-1].split(".")[0]
            pb.set_postfix_str(f"|{name}")

            embeddings_path=f"{base_embeddings_path}/{name}.npz"
            assert path.exists(embeddings_path)
            npz = np.load(embeddings_path, allow_pickle=True)
            embeddings = list(zip(npz["arr_0"], npz["arr_1"], npz["arr_2"]))

            for i in range(len(prototypes)):
                eval_dir_fold = f"{base_embeddings_path}/init/{i}"
                scores = {1:[], -1:[]}

                # Generate .CSV files with detections for FROC evaluation using CAMELYON16 lib
                with open(f"{eval_dir_fold}/{name}.csv", "w+") as f:
                    writer = csv.writer(f)
                    writer.writerow(["p", "x", "y"])
                    for (x,y), anomaly, embedding in embeddings:
                        score = cosine_similarity(prototypes[i], embedding)
                        scores[anomaly].append(score)
                        writer.writerow([score, x+patch_size/2, y+patch_size/2])

                all_scores[i][1].extend(scores[1])
                all_scores[i][-1].extend(scores[-1])


    print("GENERATING FINAL RESULTS")
    # Compute results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    aurocs, auprs, fprs, tprs, y_trues, y_scores = zip(*[get_evaluation_data(all_scores[i][-1], all_scores[i][1]) for i in range(len(prototypes))])
    name = "w/o PP"
    plot_mean_curve(ax1, aurocs, fprs, tprs, name, color="blue", linestyle="solid")
    plot_mean_pr_curve(ax2, auprs, y_trues, y_scores, name, color="blue", linestyle="solid")

    title = "CAMELYON16"
    ax1.set_xlabel("False Positive Rate", fontsize="x-large")
    ax1.set_ylabel("True Positive Rate", fontsize="x-large")
    ax1.set_title("Receiver Operating Characteristic", fontsize="x-large")

    ax2.set_xlabel("Recall", fontsize="x-large")
    ax2.set_ylabel("Precision", fontsize="x-large")
    ax2.set_title("Precision-Recall", fontsize="x-large")

    ax1.legend(loc="lower right", fontsize="large")
    ax2.legend(loc="lower left", fontsize="large")

    plt.show()
    fig.savefig(f"{base_embeddings_path}/patch_level_{title}")

    params = sum(p.numel() for p in model.parameters())
    results = {i: dict(auroc=aurocs[i], aupr=auprs[i]) for i in range(len(prototypes))}
    results["NumParams"] = params
    with open(f"{base_embeddings_path}/results_per_prototype_{cv_size}.json", "w") as f:
        json.dump(results, f, indent=2)

    pprint(results)

    print(f"Result figure generated in {base_embeddings_path}/patch_level_{title}.png")


def evaluate_slide_level_camelyon_cv(model,
                                     model_name,
                                     transform,
                                     annotations_path,
                                     evaluation_slides,
                                     slides_for_proto,
                                     cv_size,
                                     patch_size,
                                     resolution=20,):
    base_embeddings_path = f"./evaluation_results/{model_name}"
    makedirs(base_embeddings_path, exist_ok=True)

    project_evaluation_slides(model,
                              base_embeddings_path,
                              annotations_path,
                              evaluation_slides,
                              patch_size,
                              resolution,
                              transform)

    prototypes = get_prototypes(model,
                                transform,
                                slides_for_proto,
                                cv_size, patch_size,
                                base_embeddings_path)

    print("COMPUTING SLIDE-LEVEL SCORES")

    slide_scores = [{label:[] for label in evaluation_slides.keys()} for _ in range(len(prototypes))]

    for label, slides in evaluation_slides.items():
        pb = tqdm(slides, desc=label)
        for slide in pb:
            name = slide.split("/")[-1].split(".")[0]
            pb.set_postfix_str(f"|{name}")

            embeddings_path=f"{base_embeddings_path}/{name}.npz"
            assert path.exists(embeddings_path)
            npz = np.load(embeddings_path, allow_pickle=True)
            embeddings = list(zip(npz["arr_0"], npz["arr_1"], npz["arr_2"]))

            for i in range(len(prototypes)):
                scores = [cosine_similarity(prototypes[i], embedding) for _, _, embedding in embeddings]
                slide_scores[i][label].append(np.max(scores))

    print("GENERATING FINAL RESULTS")
    fig, ax = plt.subplots(1, 1, figsize=(5,5))

    aurocs, fprs, tprs = zip(*[get_roc_info(scores["normal"], scores["patho"]) for scores in slide_scores])
    plot_mean_curve(ax, aurocs, fprs, tprs, model_name, color="blue", linestyle="solid")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Slide-Level ROC: {model_name}",
    )
    ax.legend(loc="lower right")
    plt.show()
    fig.savefig(f"{base_embeddings_path}/slide_level")

    print(f"Result figure generated in {base_embeddings_path}/slide_level.png")
