import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
label_map = {}

# 1. SCAN DIRECTORIES
# ----------------------------------------
print("Scanning data directory...")
class_names = sorted(os.listdir(DATA_DIR))

# Filter out non-folders (like .DS_Store or hidden files)
valid_classes = []
for name in class_names:
    if os.path.isdir(os.path.join(DATA_DIR, name)) and not name.startswith('.'):
        valid_classes.append(name)

print(f"Found {len(valid_classes)} classes: {valid_classes}")
print("Starting Feature Extraction... This will take time.")
print("-" * 50)

# Create label map
for idx, name in enumerate(valid_classes):
    label_map[name] = idx

# 2. PROCESS IMAGES
# ----------------------------------------
for idx, class_name in enumerate(valid_classes):
    class_dir = os.path.join(DATA_DIR, class_name)
    img_files = os.listdir(class_dir)
    total_imgs = len(img_files)

    print(f"[{idx + 1}/{len(valid_classes)}] Processing '{class_name}' ({total_imgs} images)")

    for i, img_path in enumerate(img_files):
        # VISUAL PROGRESS BAR
        # Updates every 100 images to avoid slowing down the script with print statements
        if i % 100 == 0 or i == total_imgs - 1:
            progress = (i + 1) / total_imgs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f'\r    Progress: |{bar}| {int(progress * 100)}% ({i + 1}/{total_imgs})')
            sys.stdout.flush()

        img_full_path = os.path.join(class_dir, img_path)

        # Read image
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract Landmarks
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for j in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[j].x
                    y = hand_landmarks.landmark[j].y
                    x_.append(x)
                    y_.append(y)

                for j in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[j].x
                    y = hand_landmarks.landmark[j].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(label_map[class_name])

    print()  # Newline after each class finishes

# 3. TRAIN MODEL
# ----------------------------------------
print("-" * 50)
print("Feature extraction complete.")
print(f"Total samples collected: {len(data)}")
print("Training Random Forest Classifier (this step is quick)...")

# Save the label mapping
with open('label_map.pickle', 'wb') as f:
    pickle.dump(label_map, f)

# Convert to numpy
data = np.asarray(data)
labels = np.asarray(labels)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"Model Accuracy: {score * 100:.2f}%")

# Save Model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("SUCCESS: Model saved as 'model.p'. You can now run inference.py!")