import time #to track the time taken for training
from ultralytics import YOLO

model= YOLO("yolov8n.pt") #load a pretrained model

print("Start training")
start=time.time()

model.train(
      data="data/data.yaml",   #coming from Kaggle Dataset, labelled and marked for each training image
      epochs=50,
      imgsz=640,
      batch=16,
      device='mps', #specifically for Mac Mini M4 chip, force trainign to GPU 
      name="pipeline_defect_50ep"
  )

elapsed = time.time() - start
print(f"\nTraining time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")