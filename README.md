# Att-DARTS: Differentiable Neural Architecture Search for Attention

The PyTorch implementation of Att-DARTS: Differentiable Neural Architecture Search for Attention.

The codes are based on https://github.com/dragen1860/DARTS-PyTorch.

## Requirements

* Python == 3.7
* PyTorch == 1.0.1
* torchvision == 0.2.2
* pillow == 6.2.1
* numpy
* graphviz
* requests
* tqdm

We recommend downloading PyTorch from [here](https://pytorch.org/get-started/previous-versions/#v101).

<!-- If you use pipenv, simply run:
```
pipenv install
```

Or, using pip:
```
pip install -r requirements.txt
``` -->

## Datasets

* CIFAR-10/100: automatically downloaded by torchvision to `data` folder.
* ImageNet (ILSVRC2012 version): manually downloaded following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Results

### CIFAR

|           |        CIFAR-10        |        CIFAR-100        | Params(M) |
|:----------|:----------------------:|:-----------------------:|:---------:|
| DARTS     |  2.76   &plusmn; 0.09  |  16.69   &plusmn; 0.28  |    3.3    |
| Att-DARTS | **2.54** &plusmn; 0.10 | **16.54** &plusmn; 0.40 |  **3.2**  |

### ImageNet

|           |  top-1   |  top-5  | Params(M) |
|:----------|:--------:|:-------:|:---------:|
| DARTS     |   26.7   |   8.7   |    4.7    |
| Att-DARTS | **26.0** | **8.5** |  **4.6**  |

## Usage

### Architecture search (using small proxy models)

Our script occupies all available GPUs. Please set environment `CUDA_VISIBLE_DEVICES`.

To carry out architecture search using 2nd-order approximation, run:

```sh
python train_search.py --unrolled
```

The found cell will be saved in `genotype.json`.
Our resultant `Att_DARTS` is written in [genotypes.py](genotypes.py).

Inserting an attention at other locations is supported through the `--location` flag.
The locations are specified at `AttLocation` in [model_search.py](model_search.py).

### Architecture evaluation (using full-sized models)

To evaluate our best cells by training from scratch, run:

```sh
python train_CIFAR10.py --auxiliary --cutout --arch Att_DARTS  # CIFAR-10
python train_CIFAR100.py --auxiliary --cutout --arch Att_DARTS  # CIFAR-100
python train_ImageNet.py --auxiliary --arch Att_DARTS  # ImageNet
```

Customized architectures are supported through the `--arch` flag once specified in [genotypes.py](genotypes.py).

Also, you can designate the search result in `.json` through the `--arch_path` flag:

```sh
python train_CIFAR10.py --auxiliary --cutout --arch_path ${PATH}  # CIFAR-10
python train_CIFAR100.py --auxiliary --cutout --arch_path ${PATH}  # CIFAR-100
python train_ImageNet.py --auxiliary --arch_path ${PATH}  # ImageNet
```

where `${PATH}` should be replaced by the path to the `.json`.

The trained model is saved in `trained.pt`.
After training, the test script automatically runs.

Also, you can always test the `trained.pt` as indicated below.

### Test (using full-sized pretrained models)

To test a pretrained model saved in `.pt` , run:

```sh
python test_CIFAR10.py --auxiliary --model_path ${PATH} --arch Att_DARTS  # CIFAR-10
python test_CIFAR100.py --auxiliary --model_path ${PATH} --arch Att_DARTS  # CIFAR-100
python test_imagenet.py --auxiliary --model_path ${PATH} --arch Att_DARTS  # ImageNet
```

where `${PATH}` should be replaced by the path to `.pt`.

You can designate our pretrained models ([cifar10_att.pt](cifar10_att.pt), [cifar100_att.pt](cifar100_att.pt), [imagenet_att.pt](imagenet_att.pt)) or the saved `trained.pt` in [Architecture Evaluation](#architecture-evaluation-using-full-sized-models).

Also, we support customized architectures specified in [genotypes.py](genotypes.py) through the `--arch` flag, or architectures specified in `.json` through the `--arch_path` flag.

### Visualization

You can visualize the found cells in [genotypes.py](genotypes.py).
For example, you can visualize `Att-DARTS` running:

```sh
python visualize.py Att_DARTS
```

Also, you can visualize the saved cell in `.json`:

```sh
python visualize.py genotype.json
```

## Related Work

### Attention modules

This repository includes the following attentions:

* Squeeze-and-Excitation
  ([paper](https://arxiv.org/abs/1709.01507) / [code (unofficial)](https://github.com/moskomule/senet.pytorch))
* Gather-Excite
  ([paper](https://arxiv.org/abs/1810.12348) / [code (unofficial)](https://github.com/BayesWatch/pytorch-GENet))
* BAM
  ([paper](https://arxiv.org/abs/1807.06514) / [code](https://github.com/Jongchan/attention-module))
* CBAM
  ([paper](https://arxiv.org/abs/1807.06521) / [code](https://github.com/Jongchan/attention-module))
* A<sup>2</sup>-Nets
  ([paper](https://arxiv.org/abs/1810.11579) / [code (unofficial)](https://github.com/gjylt/DoubleAttentionNet))

## Reference

```bibtex
@inproceedings{att-darts2020IJCNN,
author = {Nakai, Kohei and Matsubara, Takashi and Uehara, Kuniaki},
booktitle = {The International Joint Conference on Neural Networks (IJCNN)},
title = {{Att-DARTS: Differentiable Neural Architecture Search for Attention}},
year = {2020}
}
```
