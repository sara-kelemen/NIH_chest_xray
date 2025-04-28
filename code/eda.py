# libraries
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np

df = pd.read_csv('..\\data\\Data_Entry_2017.csv')
df.head()

image_folders = [f'../data/images_{str(i).zfill(3)}' for i in range(1, 13)]

df.shape

label_counts = {}

# Loop through each row in the Finding Labels
for labels in df['Finding Labels']:
    individual_labels = labels.split('|')  # split on '|'
    for label in individual_labels:
        if label in label_counts:
            label_counts[label] += 1  # add 1 if already counted
        else:
            label_counts[label] = 1   # start counter at 1 if first time

label_df = pd.DataFrame(list(label_counts.items()), columns=['Finding', 'Count'])
label_df = label_df.sort_values('Count', ascending=False)

plt.figure(figsize=(10,6))
plt.bar(label_df['Finding'], label_df['Count'])
plt.title('Distribution of Single Findings in Chest X-ray Dataset')
plt.xlabel('Finding')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

def find_image(image_name: str) -> str:
    for folder in image_folders:
        full_path = f'{folder}/images/{image_name}' 
        if os.path.exists(full_path):
            return full_path
    print(f"Image {image_name} not found!")
    return None

# random sample 9 images for 3x3 grid
rows = []
for id, row in df.iterrows():
    path = find_image(row['Image Index'])
    if path is not None:
        rows.append(row)

df_sampling = pd.DataFrame(rows)
sample_rows = df_sampling.sample(9)

# plotting
plt.figure(figsize=(12, 12))
for idx, (_, row) in enumerate(sample_rows.iterrows()):
    img_path = find_image(row['Image Index'])
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        plt.subplot(3, 3, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(row['Finding Labels'], fontsize=8)
        plt.axis('off')
plt.suptitle('Random Sample of Chest X-rays', fontsize=16)
plt.tight_layout()
plt.show()

heights = []
widths = []
sample2 = df_sampling.sample(1000)
for id, row in sample_rows.iterrows():
    img_path = find_image(row['Image Index'])
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        heights.append(h)
        widths.append(w)

print("Most common height:", np.bincount(heights).argmax())
print("Most common width:", np.bincount(widths).argmax())

# 1. Find healthy images
healthy_df = df_sampling[df_sampling['Finding Labels'] == 'No Finding']

# 2. Find diseased images (anything that's NOT "No Finding")
diseased_df = df_sampling[df_sampling['Finding Labels'] != 'No Finding']

# 3. Randomly sample
healthy_samples = healthy_df.sample(6)
diseased_samples = diseased_df.sample(6)

# 4. Plot
plt.figure(figsize=(18,6))

# Plot healthy
for idx, (_, row) in enumerate(healthy_samples.iterrows()):
    img_path = find_image(row['Image Index'])
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        plt.subplot(2, 6, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title('Healthy', fontsize=8)
        plt.axis('off')

# Plot diseased
for idx, (_, row) in enumerate(diseased_samples.iterrows()):
    img_path = find_image(row['Image Index'])
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        plt.subplot(2, 6, idx+7)  # 6 slots for healthy first, now start at 7
        plt.imshow(img, cmap='gray')
        plt.title('Diseased', fontsize=8)
        plt.axis('off')

plt.suptitle('Comparison of Healthy vs Diseased Chest X-rays', fontsize=16)
plt.tight_layout()
plt.show()