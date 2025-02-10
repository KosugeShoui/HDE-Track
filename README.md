## HDE-Track: Sperm Detection and Tracking Model Using HDE Transformer with Spatio-Temporal Deformable Attention for Sperm Analysis Automation

<div style="align: center">
<img src=./assets/transtrack.png/>
</div>


## Introduction
editing now

## Updates
- update yet

## VISEM-Tracking Evaluation
<div style="align: center">
<img src=./assets/result.png/>
</div>


# Base Tracking Model
If download link is invalid, models and logs are also available in [Github Release](https://github.com/PeizeSun/TransTrack/releases/tag/v0.1) and [Baidu Drive](https://pan.baidu.com/s/1dcHuHUZ9y2s7LEmvtVHZZw) by code m4iv.

#### Notes
- editing now


## Demo
<img src="assets/MOT17-11.gif" width="400"/>  <img src="assets/MOT17-04.gif" width="400"/>


## Installation
The codebases are built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [CenterTrack](https://github.com/xingyizhou/CenterTrack).

#### Requirements
- Linux, CUDA>=9.2, GCC>=5.4
- Python>=3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


#### Steps
1. Install and build libs
```
git clone https://github.com/PeizeSun/TransTrack.git
cd TransTrack
cd models/ops
python setup.py build install
cd ../..
pip install -r requirements.txt
```

2. Prepare datasets and annotations
```
mkdir crowdhuman
cp -r /path_to_crowdhuman_dataset/CrowdHuman_train crowdhuman/CrowdHuman_train
cp -r /path_to_crowdhuman_dataset/CrowdHuman_val crowdhuman/CrowdHuman_val
mkdir mot
cp -r /path_to_mot_dataset/train mot/train
cp -r /path_to_mot_dataset/test mot/test
```
CrowdHuman dataset is available in [CrowdHuman](https://www.crowdhuman.org/). 
```
python3 track_tools/convert_crowdhuman_to_coco.py
```
MOT dataset is available in [MOT](https://motchallenge.net/).
```
python3 track_tools/convert_mot_to_coco.py
```

3. Pre-train on crowdhuman
```
sh track_exps/crowdhuman_train.sh
python3 track_tools/crowdhuman_model_to_mot.py
```
The pre-trained model is available [crowdhuman_final.pth](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing).

4. Train TransTrack
```
sh track_exps/crowdhuman_mot_trainhalf.sh
```

5. Evaluate TransTrack
```
sh track_exps/mot_val.sh
sh track_exps/mota.sh
```

6. Visualize TransTrack
```
python3 track_tools/txt2video.py
```


## Test set
Pre-training data | Fine-tuning data | Training time | MOTA% | FP | FN | IDs
:---:|:---:|:---:|:---:|:---:|:---:|:---:
crowdhuman | mot17 | ~40h + 2h | 68.4 | 22137  | 152064  | 3942  
crowdhuman | crowdhuman + mot17 | ~40h + 6h | 74.5 | 28323 | 112137 | 3663 

#### Notes
- Performance on test set is evaluated by [MOT challenge](https://motchallenge.net/).
- (crowdhuman + mot17) is training on mixture of crowdhuman and mot17.
- We won't release trained models for test test. Running as in Steps could reproduce them. 
 
#### Steps
1. Train TransTrack
```
sh track_exps/crowdhuman_mot_train.sh
```

or

1. Mix crowdhuman and mot17
```
mkdir -p mix/annotations
cp mot/annotations/val_half.json mix/annotations/val_half.json
cp mot/annotations/test.json mix/annotations/test.json
cd mix
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
cd ..
python3 track_tools/mix_data.py
```
2. Train TransTrack
```
sh track_exps/crowdhuman_plus_mot_train.sh
```


## License

TransTrack is released under MIT License.


## Citing

If you use TransTrack in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

