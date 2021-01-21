## Installation 
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Usage
You will need to download and prepare the COCO dataset. It is recommended to symlink the dataset root to `path_to_ltm/data`.

```bash
cd path_to_ltm
mkdir data
ln -s /path_to_coco data/coco
```

## Training example

```bash
cd path_to_ltm
tools/dist_train.sh configs/LTM/ltm_af_r50_fpn_1x.py 8 --autoscale-lr
```

For more details, please refer to the mmdetection [README.md](MMDET_README.md)

## Citations
Please consider citing our paper in your publications if the project helps your research.
```
@article{zhang2021ltm,
  author={X. {Zhang} and F. {Wan} and C. {Liu} and X. {Ji} and Q. {Ye}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning to Match Anchors for Visual Object Detection}, 
  year={2021},
  doi={10.1109/TPAMI.2021.3050494}
}
```
