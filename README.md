# Solar Panel Defect Detection with YOLO11s + SEAM + DSCA

This repository provides the implementation of the solar panel defect detection method described in the paper.



***

## Overview



* **Base model**: YOLO11s

* **Improvements**:


  * **SEAM** (Sobel-Enhanced Attention Module) is inserted at the P3/P4/P5 outputs of the Neck.

  * **DSCA** (Dual-branch Spatial-Channel Attention) is inserted after SEAM at each scale.

* **Strategy**: edge-aware spatial focusing first, then dual-branch spatial-channel enhancement.

* **Dataset**: 6,079 visible-light images of solar panels with three classes: broken, dirty, clean.

* **Test set results**: mAP@0.5 = 97.7%, mAP@0.5:0.95 = 97.7%.



***

## Environment



* Python 3.11

* PyTorch 2.11.0

* Ultralytics 8.4.12

* See requirements.txt for the full list of dependencies.



***

## Data Preparation



1. Download the dataset and organize it as follows:



```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```



1. Modify the path field in data.yaml to point to your dataset directory.



***

## Training

Run the following command to train the model with the same configuration as in the paper:



```
python test.py --data data.yaml --epochs 120 --batch 8 --imgsz 640 --device 0
```

**Note**: If you encounter GPU memory limitations, you may reduce the batch size (e.g., --batch 4). However, the default setting in the code is batch=8, which matches the paper.



***

## Test Evaluation

The script automatically evaluates the best model on the test set after training completes, so no additional command is required. It loads best.pt, runs evaluation on the test set (609 images), and prints the metrics.



***

## Overall Architecture

The framework follows the edge-aware spatial focusing first, then dual-branch spatial-channel enhancement strategy. SEAM is inserted at each Neck output, and DSCA is inserted after SEAM at the same scale.



***

## Results

All reported metrics are obtained on the test set using the best checkpoint. The model achieves:



* mAP@0.5: 97.7%

* mAP@0.5:0.95: 97.7%

For detailed per-class results and ablation studies, please refer to the paper.



***

## Citation

If you use this code in your research, please cite the original paper.



***

## License

This project is licensed under the AGPL-3.0 License, due to the dependency on Ultralytics. See the LICENSE file for details.