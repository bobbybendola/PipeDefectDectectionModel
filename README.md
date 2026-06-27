# Pipeline Defect Detection Model

**Inspired from Aditi Varia, Shrujan Sriram, Abhinav Gondesi's work on PipeDown - a tool that can be used to constantly monitor the inside of pipes for industrial pipe inspections. 
**Created during UC Berkeley AI Hackathon 2026. 
DevPost: https://devpost.com/software/tbd-nprbw3 
GitHub: https://github.com/aditivv/berkai 


A YOLOv8n object detection model trained to identify structural defects in
pipeline inspection footage. Built, trained, and run entirely locally on an
Apple M4 Mac mini using GPU acceleration. 

Trained using publically avaible Kaggle data set, with images collected from Robots driven through industrial pipes underground. 
Mir Md Musleh Uddin Shaeek. (2024). Pipeline Defect Dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/5294466 
https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset/data 


---

## Demo — Single Epoch Training on Apple M4

[![YOLOv8n Training on Apple M4]: https://youtu.be/AaJa3gm1w9M 

*Single epoch training run on Apple M4 Mac mini — ~66 seconds using MPS

---

## Defect Classes

The model detects 6 categories of pipeline defects:

| ID | Class | Description |
|----|-------|-------------|
| 0 | Deformation | Structural deformation of the pipe wall |
| 1 | Obstacle | Foreign object blocking the pipeline |
| 2 | Rupture | Break or fracture in the pipe |
| 3 | Disconnect | Pipe joint separation |
| 4 | Misalignment | Sections of pipe out of alignment |
| 5 | Deposition | Buildup of material inside the pipe |

---

## Dataset

- **Source:** [Pipeline Defect Dataset](https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset) — Kaggle
- **Total images:** 22,120 (only a training split provided by the dataset creator)
- **Subset used:** 1,000 images randomly sampled with seed 42
- **Split:** 800 train / 100 val / 100 test (handled manually — see `prepare_data.py`)
- **Label format:** YOLO format — one `.txt` file per image containing normalised
  bounding box coordinates: `class_id center_x center_y width height`

### Why Only 1,000 Images?

The full dataset is 22,120 images. A 1,000-image subset was chosen as an initial
baseline to test training viability on local hardware and establish timing benchmarks
before committing to a longer run.

---

## Hardware — Apple M4 Mac mini

| Spec | Detail |
|------|--------|
| Chip | Apple M4 |
| Memory | 16GB unified memory |
| GPU | 10-core (via Metal/MPS) |
| Storage | Local SSD |

### Why Apple Silicon Is Well Suited for This

On a traditional machine, CPU RAM and GPU VRAM are separate, the data must be
transferred between them during training, creating a bottleneck. On Apple Silicon,
the CPU and GPU share the same physical memory pool unified memory. This means
the GPU can access training data directly with zero transfer overhead, which is
why the M4 Mac mini performs well for local ML training despite having no
dedicated GPU.

I manually adjsted the Ultralytics model to be have it backpropgation and training phase tensor calclation to happen on the GPU side of compute
instead of the CPU, since Apple's MPS backend allows us to utlize the benefit of parralelize the matrix compute, instead of having these operations 
be done without parralization on the CPU itself. 

---

## Model

- **Architecture:** YOLOv8n (nano) — the smallest and fastest YOLOv8 variant
- **Parameters:** 3,012,018
- **Compute:** 8.2 GFLOPs
- **Pretrained weights:** COCO dataset (`yolov8n.pt`) — fine-tuned on pipeline defects
- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)

---

## Training Configuration

```python
model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device='mps',   #helps push compute for the model to the GPU. 
    name="pipeline_defect_50ep"
)

┌───────────┬───────┬──────────────────────────────────────────────────────────────────────┐
│ Parameter │ Value │                                Reason                                │
├───────────┼───────┼──────────────────────────────────────────────────────────────────────┤
│ epochs    │ 50    │ Full training run after 1-epoch timing baseline                      │
├───────────┼───────┼──────────────────────────────────────────────────────────────────────┤
│ imgsz     │ 640   │ Standard YOLOv8 input resolution                                     │
├───────────┼───────┼──────────────────────────────────────────────────────────────────────┤
│ batch     │ 16    │ Keeps GPU memory at ~4.3GB, well within the 16GB unified memory pool │
├───────────┼───────┼──────────────────────────────────────────────────────────────────────┤
│ device    │ mps   │ Explicitly routes computation to the M4 GPU via Apple Metal          │
└───────────┴───────┴──────────────────────────────────────────────────────────────────────┘

Why Batch Size 16 and Not 32?

During training, the GPU consumed 4.3GB of unified memory at batch size 16.
Increasing to batch size 32 would roughly double that to ~8–9GB. Since macOS and
background processes also share the same 16GB memory pool, keeping GPU usage at
4.3GB leaves comfortable headroom and avoids memory pressure causing slowdowns.

---
Results

Single Epoch Baseline

┌───────────────┬─────────────┐
│    Metric     │    Value    │
├───────────────┼─────────────┤
│ Training time │ ~66 seconds │
├───────────────┼─────────────┤
│ Precision     │ 86.7%       │
├───────────────┼─────────────┤
│ Recall        │ 6.9%        │
├───────────────┼─────────────┤
│ mAP50         │ 11.3%       │
├───────────────┼─────────────┤
│ mAP50-95      │ 4.5%        │
└───────────────┴─────────────┘

High precision but near-zero recall after 1 epoch is expected — the model had
only seen each image once and was being extremely conservative.

50 Epoch Full Run

Total training time: 34.8 minutes on Apple M4

┌────────────────┬─────────┬──────────┬──────────┬──────────┐
│     Metric     │ Epoch 1 │ Epoch 10 │ Epoch 25 │ Epoch 50 │
├────────────────┼─────────┼──────────┼──────────┼──────────┤
│ mAP50          │ 0.113   │ 0.321    │ 0.371    │ 0.482    │
├────────────────┼─────────┼──────────┼──────────┼──────────┤
│ Recall         │ 0.069   │ 0.331    │ 0.405    │ 0.454    │
├────────────────┼─────────┼──────────┼──────────┼──────────┤
│ Precision      │ 0.867   │ 0.384    │ 0.399    │ 0.552    │
├────────────────┼─────────┼──────────┼──────────┼──────────┤
│ train/cls_loss │ 3.701   │ 2.428    │ 1.766    │ 0.861    │
├────────────────┼─────────┼──────────┼──────────┼──────────┤
│ train/box_loss │ 1.656   │ 1.571    │ 1.299    │ 0.876    │
└────────────────┴─────────┴──────────┴──────────┴──────────┘

Accuracy Summary

┌───────────┬─────────────┬────────────────────────────────────────────────────────────┐
│  Metric   │ Final Value │                       Plain English                        │
├───────────┼─────────────┼────────────────────────────────────────────────────────────┤
│ mAP50     │ 48.2%       │ Finds and correctly identifies defects about half the time │
├───────────┼─────────────┼────────────────────────────────────────────────────────────┤
│ mAP50-95  │ 26.6%       │ Stricter score with tighter bounding box requirements      │
├───────────┼─────────────┼────────────────────────────────────────────────────────────┤
│ Precision │ 55.2%       │ When it flags a defect, it's correct 55% of the time       │
├───────────┼─────────────┼────────────────────────────────────────────────────────────┤
│ Recall    │ 45.4%       │ Of all actual defects present, it finds 45% of them        │
└───────────┴─────────────┴────────────────────────────────────────────────────────────┘

Against industry benchmarks:
- < 50% mAP50 — Experimental / proof of concept
- 50–70% mAP50 — Usable for assisted inspection ← approaching this range
- > 70% mAP50 — Production grade

Saved Model Weights

Ultralytics automatically saves two checkpoints:

runs/detect/pipeline_defect_50ep/weights/best.pt   ← best mAP50 across all epochs
runs/detect/pipeline_defect_50ep/weights/last.pt   ← final epoch state

---
Setup & Usage

1. Clone and set up environment

git clone <your-repo-url>
cd PipeDefectDectectionModel
python3 -m venv venv
source venv/bin/activate
pip install ultralytics

2. Prepare dataset

Download the dataset from Kaggle and place the zip in ~/Downloads, then:

python3 prepare_data.py

This samples 1,000 images from the full dataset and creates the train/val/test split.

3. Train

python3 train.py

4. Run inference with the saved model

from ultralytics import YOLO
model = YOLO("runs/detect/pipeline_defect_50ep/weights/best.pt")
results = model("path/to/your/image.jpg")
results[0].show()

---
Next Steps

- Train on a larger subset (3,000–5,000 images) to push mAP50 past 0.65
- Evaluate per-class performance on the 100-image test set
---
