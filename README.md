# Self-Supervised Deep Metric Learning for Prototypical Zero-shot Lesion Retrieval in Placenta Whole-Slide Images

Official repository for *Self-Supervised Deep Metric Learning for Prototypical Zero-shot Lesion Retrieval in Placenta Whole-Slide Image*.

[Open access link](https://www.sciencedirect.com/science/article/pii/S0010482525009850) to the article.

This repository contains the training pipeline presented in the paper, the evaluation code for CAMELYON16 and a demo notebook.

Model weights and a sample support set for CAMELYON16 can be downloaded from [this repository](https://drive.google.com/drive/folders/1H6A4JS3D06DqE4QfWbEb1Sx1pQk8wBtz).

### Installation
You can install most requirements from `requirements.txt`.
```
pip install -r requirements.txt
```

You will also need to install ASAP and the multiresolutionimageinterface library to handle CAMELYON16 annotations.

### Training

An example configuration file is available in [`configs/local_train.conf`](configs/local_train.conf). We advise you to read through the [`lib/parser.py`](lib/parser.py) file to see the different training parameters available. The training WSIs should be listed in a .json file as shown in [`./camelyon_normal_split.json`](./camelyon_normal_split.json). When ready, start training with:
```
python train.py -c configs/local_train.conf
```

### Evaluation
The patch- and slide-level zero-shot evaluation pipeline can be run from the `eval.py` file. The model_dict variable can be modified to evaluate other model/transforms. The slides to be used for prototype definition should be listed in a .json file as shown in [`./camelyon_normal_split.json`](./camelyon_normal_split.json). To run patch-level evaluation, run:
```
python eval.py --model protossdml --name protossdml_camelyon16 --level patch
```

### Demo notebook
The [`demo.ipynb`](./demo.ipynb) notebook contains an example of heatmap generation for a test WSI from CAMELYON16 using a support set generated during the zero-shot, simulated low-data regime evaluation.

<img src="./gt_test_071.png" height=300>  <img src="./hm_test_071.png" height=300>

### Citing this article
If you find this repository useful, please consider giving a star ⭐ and cite the original article!

```
@article{protossdml_2025,
	title = {Self-supervised deep metric learning for prototypical zero-shot lesion retrieval in placenta whole-slide images},
	volume = {196},
	issn = {0010-4825},
	url = {https://www.sciencedirect.com/science/article/pii/S0010482525009850},
	doi = {https://doi.org/10.1016/j.compbiomed.2025.110634},
	journal = {Computers in Biology and Medicine},
	author = {Faure, Gaspar and Soglio, Dorothée Dal and Patey, Natalie and Oligny, Luc and Girard, Sylvie and Séoud, Lama},
	year = {2025},
	pages = {110634},
}
```
