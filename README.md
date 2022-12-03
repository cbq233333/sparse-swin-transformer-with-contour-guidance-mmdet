# sparse-swin-transformer-with-contour-guidance-mmdet
A Local Sparse Information Aggregation Transformer with Explicit Contour Guidance for SAR Ship Detection
## Introduction
We propose a local-sparse-information-aggregation transformer with explicit contour guidance for ship detection in SAR images. Based on the Swin Transformer architecture, in order to effectively aggregate sparse meaningful cues of small-scale ships, a deformable attention mechanism is incorporated to change the original self-attention mechanism. Moreover, a novel contour-guided shape-enhancement module is proposed to explicitly enforce the contour constraints on the one-dimensional transformer architecture. Experimental results show that our proposed method achieves superior performance on the challenging HRSID and SSDD datasets.
## Installation
```conda create -n your_envs_name python=3.7 -y```
download torch and torchvision from [download](https://download.pytorch.org/whl/torch_stable.html)

