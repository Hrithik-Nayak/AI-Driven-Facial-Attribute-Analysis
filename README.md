detection
------------------------
from ultralytics import YOLO

# Load pre-trained YOLO26m weights
model = YOLO('yolov8m.pt')  # you can switch to YOLO26m weights if available

# Train detection
model.train(
    data='../dataset/annotations/data.yaml',
    epochs=150,            # longer training for high accuracy
    imgsz=960,             # higher resolution for small lamp detection
    batch=16,              # adjust based on GPU memory
    optimizer='AdamW',     # better for stability
    lr0=0.001,             # initial learning rate
    lr_scheduler='cosine', # smoother convergence
    name='yolo26m_lamp',
    device=0,
    augment=True,          # include mosaic, flips, color jitter
    patience=20,           # early stopping if no improvement
    save_period=5          # save checkpoints every 5 epochs
)

cropping
-------------------------------------
import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# Load trained YOLO26m
model = YOLO('../detection/yolo26m_model/best.pt')

input_folder = '../dataset/images/train'  # change per split
output_folder = '../dataset/crops/train/'

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    results = model.predict(img_path, conf=0.5)

    img = cv2.imread(img_path)

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        crop_name = f"{Path(img_name).stem}_lamp{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, crop_name), crop)



main model
----------------------------
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os

# Paths
train_dir = '../dataset/crops/train'
val_dir = '../dataset/crops/val'

# Data Augmentation
train_transforms = transforms.Compose([
    transforms.Resize((380,380)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((380,380)),
    transforms.ToTensor()
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=2)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training Loop
best_acc = 0
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_dataset)
    print(f"Validation Acc: {val_acc:.4f}")

    # Scheduler step
    scheduler.step()

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), '../classification/classifier_model/best.pt')
