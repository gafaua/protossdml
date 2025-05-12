import argparse
import json
from glob import glob
from os import path

import numpy as np
import torch

from lib.adapter import get_protossdml_encoder
from lib.evaluation import (
    evaluate_patch_level_camelyon_cv,
    evaluate_slide_level_camelyon_cv,
)

CAMELYON_DATASET_PATH = "/store2/travail/data/CAMELYON16/"

# TO RUN EVALUATION ON DIFFERENT MODELS, USE A DICT SIMILAR THE ONE BELOW

# _model_dict = {
#     "uni": get_uni,
#     "uni2": get_uni2,
#     "conch": get_conch,
#     "conch15": get_conch15,
#     "virchow2": get_virchow2,
#     "protossdml": get_protossdml,
#     "musk": get_musk_image_encoder,
# }

_model_dict = {
    "protossdml": get_protossdml_encoder
}

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(_model_dict.keys()))
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, choices=["patch", "slide"])
    args = parser.parse_args()

    print(f"RUNNING EVALUATION FOR MODEL {args.model}")

    model, transform = _model_dict[args.model]()

    # CAMELYON16
    with open("./camelyon_normal_split.json", "r") as f:
        splits = json.load(f)

    slides_for_proto = [path.join("/store2/travail/data/", s) for s in splits["proto"]]

    annotations_path = f"{CAMELYON_DATASET_PATH}/annotations/"
    evaluation_slides = dict(normal=[], patho=[])
    all_test_slides = glob(f"{CAMELYON_DATASET_PATH}/images/test*")

    for slide in all_test_slides:
        name = slide.split("/")[-1].split(".")[0]
        if glob(path.join(annotations_path, f"{name}.xml")):
            evaluation_slides["patho"].append(slide)
        else:
            evaluation_slides["normal"].append(slide)

    print(f"CAMELYON TEST, {len(evaluation_slides['normal'])} normal and {len(evaluation_slides['patho'])} patho")

    if args.level == "patch":
        evaluate_patch_level_camelyon_cv(
                                    model,
                                    args.name,
                                    transform,
                                    annotations_path,
                                    evaluation_slides,
                                    slides_for_proto,
                                    10,
                                    patch_size=384 if args.name == "musk" else 224,
        )
    else:
        evaluate_slide_level_camelyon_cv(
                                    model,
                                    args.name,
                                    transform,
                                    annotations_path,
                                    evaluation_slides,
                                    slides_for_proto,
                                    10,
                                    patch_size=384 if args.name == "musk" else 224,
        )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    evaluate()
