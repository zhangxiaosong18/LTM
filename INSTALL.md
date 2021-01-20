## Installation

### Requirements:
- Python3
- PyTorch 1.3.1 with CUDA support
- torchvision 0.4.2
- mmcv 0.2.14
- pycocotools 2.0.0


### Step-by-step installation

```bash
conda create -n ltm python=3.7
conda activate ltm
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
pip install mmcv===0.2.14 pycocotools===2.0.0

git clone https://github.com/zhangxiaosong18/LTM.git
cd LTM
python setup.py develop
```
