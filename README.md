# AIoU: An Adaptive Bounding Box Loss with an Edge Alignment Penalty

This repository is an improved version of the standard [Ultralytics](https://github.com/ultralytics/ultralytics) framework, specifically designed to incorporate **AIoU (Adaptive Intersection over Union)**, a novel bounding box regression loss function.

---

## 🚀 Overview

**AIoU** is designed to address the limitations of standard IoU variants (like CIoU) in challenging scenarios, such as detecting **small objects** or resolving **highly overlapping instances**. By introducing an **Edge Alignment Penalty** and an **Adaptive Weighting Mechanism**, AIoU provides a more granular optimization signal for precise boundary alignment.

### Key Features
- **Edge Alignment Penalty**: Explicitly measures the Euclidean distance between corresponding edge midpoints of the predicted and ground-truth boxes.
- **Adaptive Weighting**: Dynamically balances the aspect ratio penalty and the edge alignment penalty based on the IoU distribution of the batch.
- **Seamless Integration**: Fully compatible with the YOLO series (YOLOv10, YOLO11, YOLOv12) within the Ultralytics ecosystem.
- **Superior Performance**: Consistent AP gains (+0.1% to +0.4%) on COCO and VisDrone datasets without adding inference overhead.

---

## 📖 Methodology

AIoU enhances the Complete IoU (CIoU) formulation by adding a specialized edge penalty term $e$ and dynamic weights $\lambda_v, \lambda_e$:

$$AIoU = \text{IoU} - \left( \frac{\rho^2(b, b^{gt})}{c^2} + \lambda_v \cdot \alpha \cdot v + \lambda_e \cdot \gamma \cdot e \right)$$

### 1. Edge Alignment Penalty ($e$)
The edge penalty $e$ is calculated as the average normalized squared Euclidean distance between corresponding edge midpoints (left, right, top, bottom):
$$e_{raw} = \frac{1}{4} \left( \frac{d^2_{left}}{(h^{gt})^2} + \frac{d^2_{right}}{(h^{gt})^2} + \frac{d^2_{top}}{(w^{gt})^2} + \frac{d^2_{bottom}}{(w^{gt})^2} \right)$$
This term is constrained using an arctan function to ensure stability: $e = \frac{2}{\pi} \arctan(e_{raw})$.

The trade-off parameters $\alpha$ and $\gamma$ are defined as:
$$\alpha = \frac{v}{v - \text{IoU} + (1 + \epsilon)}, \quad \gamma = \frac{e}{e - \text{IoU} + (1 + \epsilon)}$$

### 2. Adaptive Weighting Mechanism
A scale factor is computed using the batch IoU mean ($\mu_{IoU}$) and interquartile range spread:
$$\text{scale\_factor} = (1 - \mu_{IoU}) \cdot (C_{spread} + \text{spread})$$
The weights $\lambda_e$ and $\lambda_v$ are then adaptively adjusted:
- $\lambda_e = \tanh(\text{scale\_factor} \times (0.5 - \text{IoU}_{pair}))$
- $\lambda_v = 1 - \lambda_e$

This allows the model to prioritize **edge alignment** for low-IoU pairs and **aspect ratio refinement** for high-IoU pairs.

---

## 📊 Performance

Experiments conducted on **MS COCO 2017** and **VisDrone 2019** demonstrate the effectiveness of AIoU across various YOLO architectures.

### COCO 2017 Results
| Model | Loss | AP(%) | $\Delta$AP | Params(M) | FLOPs(G) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv10n | CIoU | 38.5 | | 2.3 | 6.7 |
| YOLOv10n | **AIoU** | **38.9** | **+0.4** | 2.3 | 6.7 |
| YOLOv10s | CIoU | 46.2 | | 7.2 | 21.6 |
| YOLOv10s | **AIoU** | **46.5** | **+0.3** | 7.2 | 21.6 |
| YOLOv10m | CIoU | 51.0 | | 15.4 | 59.1 |
| YOLOv10m | **AIoU** | **51.3** | **+0.3** | 15.4 | 59.1 |
| YOLOv10b | CIoU | 52.5 | | 19.1 | 92.0 |
| YOLOv10b | **AIoU** | **52.8** | **+0.3** | 19.1 | 92.0 |
| YOLOv10l | CIoU | 53.2 | | 24.4 | 120.3 |
| YOLOv10l | **AIoU** | **53.4** | **+0.2** | 24.4 | 120.3 |
| YOLOv10x | CIoU | 54.4 | | 29.5 | 160.4 |
| YOLOv10x | **AIoU** | **54.7** | **+0.3** | 29.5 | 160.4 |
| YOLO11n | CIoU | 39.4 | | 2.6 | 6.5 |
| YOLO11n | **AIoU** | **39.7** | **+0.3** | 2.6 | 6.5 |
| YOLO11s | CIoU | 46.9 | | 9.4 | 21.5 |
| YOLO11s | **AIoU** | **47.2** | **+0.3** | 9.4 | 21.5 |
| YOLO11m | CIoU | 51.5 | | 20.1 | 68.0 |
| YOLO11m | **AIoU** | **51.7** | **+0.2** | 20.1 | 68.0 |
| YOLO11l | CIoU | 53.3 | | 25.3 | 86.9 |
| YOLO11l | **AIoU** | **53.6** | **+0.3** | 25.3 | 86.9 |
| YOLO11x | CIoU | 54.6 | | 56.9 | 194.9 |
| YOLO11x | **AIoU** | **54.8** | **+0.2** | 56.9 | 194.9 |
| YOLOv12n | CIoU | 40.6 | | 2.5 | 6.5 |
| YOLOv12n | **AIoU** | **41.0** | **+0.4** | 2.5 | 6.5 |
| YOLOv12s | CIoU | 48.0 | | 9.1 | 21.4 |
| YOLOv12s | **AIoU** | **48.2** | **+0.2** | 9.1 | 21.4 |
| YOLOv12m | CIoU | 52.5 | | 19.6 | 67.5 |
| YOLOv12m | **AIoU** | **52.9** | **+0.4** | 19.6 | 67.5 |
| YOLOv12l | CIoU | 53.7 | | 26.5 | 88.9 |
| YOLOv12l | **AIoU** | **53.8** | **+0.1** | 26.5 | 88.9 |
| YOLOv12x | CIoU | 55.2 | | 59.3 | 199.0 |
| YOLOv12x | **AIoU** | **55.5** | **+0.3** | 59.3 | 199.0 |

### VisDrone Results
| Model | Loss | AP(%) | $\Delta$AP | Params(M) | FLOPs(G) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv10n | CIoU | 20.1 | | 2.3 | 6.7 |
| YOLOv10n | **AIoU** | **20.4** | **+0.3** | 2.3 | 6.7 |
| YOLOv10s | CIoU | 24.2 | | 7.2 | 21.6 |
| YOLOv10s | **AIoU** | **24.5** | **+0.3** | 7.2 | 21.6 |
| YOLOv10m | CIoU | 27.6 | | 15.4 | 59.1 |
| YOLOv10m | **AIoU** | **28.0** | **+0.4** | 15.4 | 59.1 |
| YOLOv10b | CIoU | 29.1 | | 19.1 | 92.0 |
| YOLOv10b | **AIoU** | **29.4** | **+0.3** | 19.1 | 92.0 |
| YOLOv10l | CIoU | 29.7 | | 24.4 | 120.3 |
| YOLOv10l | **AIoU** | **30.1** | **+0.4** | 24.4 | 120.3 |
| YOLOv10x | CIoU | 30.5 | | 29.5 | 160.4 |
| YOLOv10x | **AIoU** | **30.9** | **+0.4** | 29.5 | 160.4 |
| YOLO11n | CIoU | 20.3 | | 2.6 | 6.5 |
| YOLO11n | **AIoU** | **20.5** | **+0.2** | 2.6 | 6.5 |
| YOLO11s | CIoU | 24.5 | | 9.4 | 21.5 |
| YOLO11s | **AIoU** | **24.8** | **+0.3** | 9.4 | 21.5 |
| YOLO11m | CIoU | 28.0 | | 20.1 | 68.0 |
| YOLO11m | **AIoU** | **28.3** | **+0.3** | 20.1 | 68.0 |
| YOLO11l | CIoU | 29.9 | | 25.3 | 86.9 |
| YOLO11l | **AIoU** | **30.3** | **+0.4** | 25.3 | 86.9 |
| YOLO11x | CIoU | 30.8 | | 56.9 | 194.9 |
| YOLO11x | **AIoU** | **31.1** | **+0.3** | 56.9 | 194.9 |
| YOLOv12n | CIoU | 20.3 | | 2.5 | 6.5 |
| YOLOv12n | **AIoU** | **20.6** | **+0.3** | 2.5 | 6.5 |
| YOLOv12s | CIoU | 24.7 | | 9.1 | 21.4 |
| YOLOv12s | **AIoU** | **24.9** | **+0.2** | 9.1 | 21.4 |
| YOLOv12m | CIoU | 28.3 | | 19.6 | 67.5 |
| YOLOv12m | **AIoU** | **28.6** | **+0.3** | 19.6 | 67.5 |
| YOLOv12l | CIoU | 30.2 | | 26.5 | 88.9 |
| YOLOv12l | **AIoU** | **30.5** | **+0.3** | 26.5 | 88.9 |
| YOLOv12x | CIoU | 30.7 | | 59.3 | 199.0 |
| YOLOv12x | **AIoU** | **31.1** | **+0.4** | 59.3 | 199.0 |

---

## 🛠️ Usage

AIoU is implemented in `ultralytics/utils/metrics.py`. To use it during training, ensure the `AIoU=True` flag is passed to the `bbox_iou` function. This has been integrated into:
- `ultralytics/utils/loss.py`: Used in `BboxLoss` for training.
- `ultralytics/utils/tal.py`: Used in `TaskAlignedAssigner` for sample assignment.

```python
# Example usage in metrics.py
from ultralytics.utils.metrics import bbox_iou

# Calculate AIoU
iou = bbox_iou(pbox, tbox, CIoU=False, AIoU=True)
loss = 1.0 - iou
```

---

## 📜 Citation

If you find AIoU useful in your research, please consider citing our work:

```latex
@article{aiou2026,
  title={AIoU: An Adaptive Bounding Box Loss with an Edge Alignment Penalty},
  author={Wenhui Chen and Ziyao Lin and Xinyu Jiang and Chi Man Vong},
  journal={IEVC},
  year={2026}
}
```

---
*Note: This project is a research-oriented modification of the Ultralytics YOLO framework.*
