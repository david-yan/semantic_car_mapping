# semantic_car_mapping
Segmentation, and semantic mapping of vehicles from LiDAR data using PointRCNN

## Data Pre-Processing
Generate data:
```
python scripts/generate_data.py --boxes data/very-important-3-cuboids.json --output data
```

Visualize data:
```
python scripts/visualize_data.py --input data/0_10.pkl
```

Labelled cuboids stored under data/cuboids
Model input data samples (.pkl frames from 3 ROS bags) stored under data/inputs

## Visualization and Evaluation of Results
Visualize model output for a single farme:
```
python scripts/visualize_model.py --model data/outputs/with_aug_model_pred/243.pkl
rviz -d rviz/vis_model.rviz
```

Evaluate model output (provide path to directory of all .pkl files):
```
python scripts/evaluate_model.py --path data/outputs/with_aug_model_pred/
```

Model output samples stored under data/outputs

## Google Drive
Full input and output zip data files stored on Google Drive: https://drive.google.com/drive/folders/1brCyyiudENFErpSPvsG-XFnR8bd8yz6r?usp=sharing


# PointRCNN
## Installation
Under poitnrcnn repo, build and install the pointnet2_lib, iou3d, roipool3d libraries by executing the following command:
```
sh build_and_install.sh
```

## Dataset
Organized the data as follow:
```
PointRCNN
├── data
│   ├── PENN
│   │   ├── PERCH
│   │   │   ├──falcon_processed.pkl
│   │   │   ├──...
├── lib
├── pointnet2_lib
├── tools
```

## Training
Under the tools repo, train the first stage of network with following command:
```
python train_perch.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200
```

Train the second stage of network with following command:
```
python train_perch.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth
```

## Evaluation
Run the evaluation script which also saves predicted boxes as pickle files under pointrcnn/output/rcnn:
```
python eval_perch.py --cfg_file cfgs/default.yaml --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_70.pth --batch_size 1 --eval_mode rcnn
```

## Citation
```
@InProceedings{Shi_2019_CVPR,
    author = {Shi, Shaoshuai and Wang, Xiaogang and Li, Hongsheng},
    title = {PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```