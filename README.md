# semantic_car_mapping
Segmentation, and semantic mapping of vehicles from LiDAR data using PointRCNN

Generate data:
```
python scripts/generate_data.py --boxes data/very-important-3-cuboids.json --output data
```

Visualize data:
```
python scripts/visualize_data.py --input data/0_10.pkl
```

Visualize model output:
```
python scripts/visualize_model.py --model data/pred_old_mode/71.pkl
rviz -d rviz/vis_model.rviz
```

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

Run the metric calculation script located under semantic_car_mapping/scripts to evaluate the model:
```
python3 evaluate_model.py --path ../pointrcnn/output/rcnn/[name of repo contained pickle files]
```
