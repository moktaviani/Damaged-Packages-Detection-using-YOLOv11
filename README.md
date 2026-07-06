# Damaged Packages Detection — YOLOv11

An object detection model for identifying **damaged packages** using **YOLOv11 (Ultralytics)**. This project is trained on a dataset from Roboflow and runs on Google Colab.

## 📋 Overview

This notebook implements a complete object detection workflow:
1. Install dependencies (`roboflow`, `ultralytics`)
2. Check GPU (CUDA) availability
3. Download the dataset from Roboflow in YOLOv11 format
4. Train a YOLOv11 model (*nano* variant) on the dataset
5. Evaluate the model (mAP50, mAP50-95) and visualize training results and the confusion matrix
6. Run inference on user-uploaded images
7. Download the best trained model (`best.pt`)

## 🗂️ Dataset

- **Source**: Roboflow
- **Workspace**: `bdata-497-advanced-topics-in-data-visualization`
- **Project**: `final-project-zseyl`
- **Version**: `2`
- **Format**: YOLOv11
- **Link**: [DamagedPackages](https://universe.roboflow.com/bdata-497-advanced-topics-in-data-visualization/final-project-zseyl)

The dataset is downloaded automatically via the Roboflow API using an API key stored in Google Colab Secrets under the name `DamagePackages_API`.

## ⚙️ Requirements

- Python 3.x
- Google Colab (recommended, since the notebook uses `google.colab.userdata` and `google.colab.files`)
- GPU (recommended to speed up training)
- Libraries:
  ```bash
  pip install roboflow ultralytics
  ```

## 🔑 API Key Setup

Before running the notebook, add your Roboflow API key to **Colab Secrets** under the name:
```
DamagePackages_API
```

## 🚀 How to Run

1. Open the `DamagedPackagesDetectionYOLOv11.ipynb` notebook in Google Colab.
2. Run the dependency installation cell.
3. Make sure the Roboflow API key has been added to Colab Secrets.
4. Run the cell that downloads the dataset from Roboflow.
5. Run the training cell:
   ```python
   from ultralytics import YOLO

   model = YOLO("yolo11n.pt")
   results = model.train(
       data=yaml_path,
       epochs=50,
       imgsz=640,
       batch=16,
       patience=15,
       project="damaged_package_runs",
       name="train",
       plots=True
   )
   ```
6. Evaluate the model on the test split to check the `mAP50` and `mAP50-95` metrics.
7. Use the prediction cell to upload new images and view the detection results.
8. Download the best model (`best.pt`) for use outside the notebook.

## 🏋️ Training Configuration

| Parameter   | Value            |
|-------------|------------------|
| Base model  | `yolo11n.pt` (YOLOv11 Nano) |
| Epochs      | 50               |
| Image size  | 640 x 640        |
| Batch size  | 16               |
| Patience    | 15 (early stopping) |
| Project dir | `damaged_package_runs` |

## 📊 Evaluation

The model is evaluated on the *test* split using the following metrics:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.5–0.95

Additional visualizations generated:
- `results.png` — training loss & metric curves
- `confusion_matrix.png` — confusion matrix from evaluation

## 🔍 Inference

The best trained model (`best.pt`) can be used to predict on new images:
```python
from ultralytics import YOLO

best_model = YOLO("path/to/best.pt")
results = best_model.predict(
    source="image.jpg",
    conf=0.25,
    save=True
)
```
