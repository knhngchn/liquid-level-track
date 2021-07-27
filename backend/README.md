# Liquid Level Tracker - Backend
Backend detection model for the tracking system.

### Train

- With validation set
```
python train.py --dataset csv --csv_train ./annotations.csv --csv_val ./val_annotations.csv --csv_classes ./classes.csv --batch_size 4 --dump_prefix 0926_ > log.txt
```

- Prepare `val_annotations.csv` and `annotations.csv`
- `dump_prefix` is the prefix of checkpoints
  - `dump_prefix 0926_`: checkpoint will be `0926_csv_retinanet_{epoch}.state_dict`

- Without validation set
```
python train.py --dataset csv --csv_train ./annotations.csv --csv_classes ./classes.csv > log.txt
```

### Test
```
python test.py --dataset csv --csv ./val_annotations.csv --csv_classes classes.csv --state_dict /path/to/your/model/weight.state_dict
```

### Visualize
```
python visualize.py --csv_classes classes.csv --img_prefix ./data/aggregation/ --state_dict /path/to/your/weight.state_dict --step_frame 100 --conf_thres 0.7 --dump_dir ./visualize_image/
```

- `img_prefix`: the prefix of path to the images
- `step_frame`: the step for visualizing
- `conf_thres`: the confidence threshold
- `dump_dir`: the directory to store visualization images

### Dependency

- python 3.6.7
- torch 0.4.1
