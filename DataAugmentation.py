import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

# Define the augmentation pipeline
transform = A.Compose([
    A.Resize(256, 256),  # Resize images to a fixed size
    A.HorizontalFlip(p=0.5),  # Random horizontal flipping
    A.VerticalFlip(p=0.3),  # Random vertical flipping
    A.RandomRotate90(p=0.5),  # Random 90-degree rotation
    A.RandomBrightnessContrast(p=0.2),  # Adjust brightness & contrast
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Elastic transformations
    A.GridDistortion(p=0.2),  # Grid distortions
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for models
    ToTensorV2()
])

# Define dataset directories
input_dir = "Dataset/data"  # Modify this to your dataset path
output_dir = "Dataset/augmented_data"
os.makedirs(output_dir, exist_ok=True)

# Augmentation settings
num_augmentations = 5  # Number of augmented copies per image

# Process images
for img_name in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i in range(num_augmentations):
        augmented = transform(image=image)['image']
        augmented = augmented.permute(1, 2, 0).numpy() * 255  # Convert back to numpy image
        augmented = cv2.cvtColor(augmented.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Save augmented image
        new_filename = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, new_filename), augmented)

print("Dataset augmentation completed!")
