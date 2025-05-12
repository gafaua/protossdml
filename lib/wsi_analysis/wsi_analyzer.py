from os import path
from typing import List

import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.augmentations import (
    CAMELYON_NORMALIZATION_MEAN,
    CAMELYON_NORMALIZATION_STD,
    MacenkoNormalizer,
)
from lib.post_processing import make_graph, propagate_scores
from lib.vision_transformer import VisionTransformer
from lib.visualizations import visualize_attention
from lib.wsi_analysis.wsi_dataset import WSIDataset


class WSIAnalyzer:
    def __init__(self,
                 model: nn.Module,
                 transform=None,
                 device="cuda",
                 ) -> None:
        self.model = model
        self.model.to(device)
        self.model.eval()

        if transform is None:
            self.transform = nn.Sequential(*[
                MacenkoNormalizer("cpu"),
                transforms.Normalize(mean=CAMELYON_NORMALIZATION_MEAN,
                                     std=CAMELYON_NORMALIZATION_STD),
            ])
        else:
            self.transform = transform

        self.device = device
        print("WSI Analyser ready to accept slides.")

    def set_wsi(self,
                wsi_path: str,
                patch_size: int,
                resolution: float,
                annotations_paths: List[str],
                annotations_type: str=None,):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.resolution = resolution
        self.annotations_type = annotations_type

        self.dataset = WSIDataset(wsi_path,
                                  patch_size,
                                  resolution,
                                  transforms=self.transform,
                                  annotations_files=annotations_paths,
                                  annotations_type=annotations_type,
                                  verbose=True)


    def preprocess(self, features_path=None):
        if features_path is None:
            self.generate_features(128)
        elif path.exists(f"{features_path}.npz"):
            npz = np.load(f"{features_path}.npz", allow_pickle=True)
            self.features = list(zip(npz["arr_0"], npz["arr_1"], npz["arr_2"]))
        else:
            self.generate_features(128)
            coords, anomalies, features = zip(*self.features)

            np.savez(features_path, coords, anomalies, features)

        print("GENERATING ADJACENCY GRAPH")
        self.generate_graph()


    @torch.no_grad()
    def generate_features(self, batch_size=128, num_workers=16):
        loader = DataLoader(self.dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
        self.features = []

        for batch in tqdm(loader, desc="Projecting patches to texture space"):
            img, anomalies, coords = batch
            features = self.model(img.to(self.device))
            features = features.cpu().numpy()
            features = np.squeeze(features)

            if len(anomalies) == 1:
                features = [features]

            coords = list(zip(*coords))
            for i in range(len(anomalies)):
                self.features.append(((coords[i][0].item(), coords[i][1].item()), anomalies[i].item(), features[i]))


    def load_support_set_from_file(self, fname):
        npz = np.load(fname, allow_pickle=True)
        self.support_set = npz["arr_0"]
        print(f"Queries shape: {self.support_set.shape}")
        self.support_set = self.support_set.tolist()


    @torch.no_grad()
    def visualize_vit_attention(self, image):
        assert isinstance(self.model, VisionTransformer)

        visualize_attention(self.model, image, self.transform, "attn")


    def initial_scoring(self):
        prototype = np.mean(self.support_set, axis=0)

        def cosine_similarity(feature):
            return 1+np.dot(prototype, feature)/(np.linalg.norm(prototype)*np.linalg.norm(feature))

        scores = []

        for coord, anomaly, feature in self.features:
            scores.append((coord, anomaly, cosine_similarity(feature)))

        self.scores = scores


    def analyse(self,):
        self.initial_scoring()
        return self.generate_score_histogram(self.scores)


    def generate_thumbnail_img(self, include_annotations):
        self.generate_scores_image([], f"gt_{self.wsi_path.split('/')[-1].split('.')[0]}", self.patch_size, include_annotations=include_annotations)


    def generate_heatmap(self, propagated, include_annotations=True):
        scores = self.pp_scores if propagated else self.scores

        self.generate_scores_image(scores,
                                   f"hm_{self.wsi_path.split('/')[-1].split('.')[0]}",
                                   self.patch_size,
                                   include_annotations=include_annotations)


    def generate_score_histogram(self,
                                 scores):

        coords, labels, scores = zip(*scores)

        fig = px.histogram(scores,
                           title="Score distribution (similarity to prototype)",
                           color=["Anomaly" if a==1 else "Normal" for a in labels],
                           marginal="box")
        return fig


    def generate_graph(self):
        coords, _, _ = zip(*self.features)

        self.graph = make_graph(np.array(coords))
        print(f"Generated graph with {len(self.graph.nodes):,} nodes and {len(self.graph.edges):,} edges")

        return self.graph


    def post_process(self, alpha, steps, threshold):
        self.pp_scores = propagate_scores(self.graph, self.scores, alpha, steps, threshold)


    def generate_scores_image(self,
                              scores,
                              name,
                              patch_size,
                              thumbnail_resolution=1.25,
                              save_image=True,
                              include_annotations=True):
        # Generate base thumbnail
        reader = self.dataset.patches.wsi
        div = self.resolution/thumbnail_resolution
        patch_size = int(patch_size / div)
        thumbnail = reader.slide_thumbnail(resolution=thumbnail_resolution,
                                           units="power")
        # Start canvas
        plt.clf()
        plt.imshow(thumbnail)
        ax = plt.gca()
        if len(scores) > 0:
            _, _, scores_ = zip(*scores)
            # This range should be adjusted accordingly
            mn, mx = min(scores_), max(scores_)
            print(mn, mx)
            dv = mx-mn
            colormap = cm.get_cmap("inferno")

            for (x, y), _, score in scores:
                x, y = int(x / div), int(y / div)
                c = colormap((score - mn)/dv)

                ax.add_patch(Rectangle((x, y), patch_size, patch_size, alpha=1, facecolor=c, lw=0.0))

        if self.dataset.annotations is not None and include_annotations:
            assert self.annotations_type == "camelyon_xml"
            div = reader.info.objective_power/thumbnail_resolution
            ann_colors = ["cyan"]
            labels = ["Tumor"]
            for label, color in zip(labels, ann_colors):
                for polygon in self.dataset.annotations[label]:
                    poly = np.array(polygon) / div
                    ax.add_patch(Polygon(poly, facecolor="none", edgecolor=color, lw=0.5))

        plt.title(name)
        plt.axis("off")

        if save_image:
            plt.savefig(name, dpi=800)

