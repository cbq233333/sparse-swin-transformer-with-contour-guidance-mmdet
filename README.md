# sparse-swin-transformer-with-contour-guidance-mmdet
A Local Sparse Information Aggregation Transformer with Explicit Contour Guidance for SAR Ship Detection
## Introduction
We propose a local-sparse-information-aggregation transformer with explicit contour guidance for ship detection in SAR images. Based on the Swin Transformer architecture, in order to effectively aggregate sparse meaningful cues of small-scale ships, a deformable attention mechanism is incorporated to change the original self-attention mechanism. Moreover, a novel contour-guided shape-enhancement module is proposed to explicitly enforce the contour constraints on the one-dimensional transformer architecture. Experimental results show that our proposed method achieves superior performance on the challenging HRSID and SSDD datasets.
## Installation
```conda create -n your_envs_name python=3.7 -y```
download torch and torchvision from [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html). We use the version of torch-1.12.1, torchvision-0.13.0 and cuda-11.6.
```pip install torch-1.12.1+cu116-cp37-cp37m-linux_x86_64.whl  
pip install torchvision-0.13.0+cu116-cp37-cp37m-linux_x86_64.whl  
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.1/index.html  
git clone https://github.com/cbq233333/sparse-swin-transformer-with-contour-guidance-mmdet.git
cd sparse-swin-transformer-with-contour-guidance-mmdet  
pip install -r requirements/build.txt
pip install -v -e . 
```
## Dataset
SSDD dataset and HRSID dataset
## Training  
```python tools/train.py configs/aa_fcos_new/deswin_conv_sample_edge.py```
## Testing
```python tools/test.py configs/aa_fcos_new/deswin_conv_sample_edge.py work_dirs/your_result_path/epoch_best.pth```  


