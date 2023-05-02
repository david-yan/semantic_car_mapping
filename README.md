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