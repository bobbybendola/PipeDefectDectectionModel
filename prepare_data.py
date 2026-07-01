import os
import random
import shutil

images_src = os.path.expanduser("~/Desktop/pipeline-detector/dataset/images/images/train")
labels_src = os.path.expanduser("~/Desktop/pipeline-detector/dataset/labels/labels/train")
output_dir = os.path.expanduser("~/Desktop/Processing/PipeDefectDectectionModel/data")

# Find all images that have a matching label file
image_files = [f for f in os.listdir(images_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
pairs = []
for img_file in image_files:
    base = os.path.splitext(img_file)[0]
    if os.path.exists(os.path.join(labels_src, base + '.txt')):
        pairs.append((img_file, base + '.txt'))

print(f"Found {len(pairs)} valid image-label pairs")

# Randomly sample 1000 , don't want to use all of the images for training 
random.seed(42)
random.shuffle(pairs)
subset = pairs[:3000]

# Split 800 train / 100 val / 100 test
# 80-10-10 split for training, testing and validation 
splits = {
    'train': subset[:2400],
    'val':   subset[2400:2700],
    'test':  subset[2700:3000],
}

# Copy files into clean structure
for split_name, files in splits.items():
    img_out = os.path.join(output_dir, 'images', split_name)
    lbl_out = os.path.join(output_dir, 'labels', split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_file, lbl_file in files:
        shutil.copy2(os.path.join(images_src, img_file), os.path.join(img_out, img_file))
        shutil.copy2(os.path.join(labels_src, lbl_file), os.path.join(lbl_out, lbl_file))

    print(f"{split_name}: {len(files)} files copied")

# Write data.yaml   
yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test
nc: 6
names:
  - Deformation
  - Obstacle
  - Rupture
  - Disconnect
  - Misalignment
  - Deposition
"""

with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("data.yaml written")
print("Done! Dataset ready at:", output_dir)