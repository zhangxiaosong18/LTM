## FreeAnchor
The Code for ["Learning to Match Anchors for Visual Object Detection"](http://www.computer.org/csdl/10.1109/TPAMI.2021.3050494.).

## Installation 
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Usage
You will need to download the COCO dataset and configure your own paths to the datasets.

```bash
cd path_to_ltm
mkdir data
ln -s /path_to_coco data/coco
```

#### Training

```bash
cd path_to_ltm
tools/dist_train.sh configs/LTM/ltm_r50_fpn_1x.py 8 --autoscale-lr
```

#### Test on COCO test-dev

```bash
cd path_to_ltm
tools/dist_test.sh configs/LTM/ltm_r50_fpn_1x.py work_dirs/ltm_r50_fpn_1x/latest.pth 8 --out results.pkl --eval bbox
```

## Citations
Please consider citing our paper in your publications if the project helps your research.
```
@ARTICLE{9321141,
  author={X. {Zhang} and F. {Wan} and C. {Liu} and X. {Ji} and Q. {Ye}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning to Match Anchors for Visual Object Detection}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3050494}}
```
