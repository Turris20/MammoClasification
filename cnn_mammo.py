import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import cv2
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def preprocess_mammo(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.bilateralFilter(img, d=7, sigmaColor=20, sigmaSpace=20)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    img = cv2.erode(img, kernel, iterations=1)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img)

def resize_with_padding(img, target_size=(512, 512)):
    target_w, target_h = target_size
    original_w, original_h = img.size

    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    new_img.paste(img, (pad_left, pad_top))
    return new_img

class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        return resize_with_padding(img, self.target_size)


class MammogramDataset(Dataset):
    def __init__(self, csv_path, mode="train", base_dir="", img_size=(512, 512)):
        self.mode = mode
        self.base_dir = base_dir
        self.img_size = img_size

        self.df = pd.read_csv(csv_path)
        self.df['label'] = self.df['classification'].str.lower().map({'benign': 0, 'malignant': 1})

        self.df['raw_image_path'] = self.df['raw_image_path'].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(self.base_dir, p)
        )

        self.df = self.df[self.df['raw_image_path'].apply(lambda p: os.path.exists(p))]

        self.img_paths = self.df['raw_image_path'].values
        self.labels = self.df['label'].values

        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]

        if mode == "train":
            self.transform = transforms.Compose([
                ResizeWithPadding(self.img_size),

                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.15),
                transforms.RandomRotation(degrees=10),

                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.12, contrast=0.18)
                ], p=0.4),

                transforms.RandomResizedCrop(self.img_size, scale=(0.86, 1.0), ratio=(0.98, 1.02)),

                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.25),

                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])
        else:
            self.transform = transforms.Compose([
                ResizeWithPadding(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = preprocess_mammo(img)
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvNeXtLarge_CBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, img_size=(512,512)):
        super().__init__()

        weights_enum = None
        backbone_fn = None
        if hasattr(models, "convnext_large"):
            backbone_fn = models.convnext_large
            if hasattr(models, "ConvNeXt_Large_Weights"):
                weights_enum = models.ConvNeXt_Large_Weights
        if backbone_fn is None:
            raise RuntimeError("Your torchvision does not have convnext_large. Update torchvision or request fallback code.")

        if pretrained and weights_enum is not None:
            weights = getattr(weights_enum, "IMAGENET1K_V1", None) or getattr(weights_enum, "DEFAULT", None)
            self.backbone = backbone_fn(weights=weights)
        else:
            self.backbone = backbone_fn(weights=None)

        self.stage_indices = []
        for i, m in enumerate(self.backbone.features):
            if isinstance(m, nn.Sequential) and len(list(m.children())) > 1:
                self.stage_indices.append(i)

        self.cbam_map = nn.ModuleDict()
        with torch.no_grad():
            x = torch.randn(1, 3, img_size[0], img_size[1])
            cur = x
            for i, module in enumerate(self.backbone.features):
                cur = module(cur)
                if i in self.stage_indices:
                    planes = cur.shape[1]
                    self.cbam_map[str(i)] = CBAM(planes=planes)

        with torch.no_grad():
            tmp = x
            for module in self.backbone.features:
                tmp = module(tmp)
            pooled = nn.functional.adaptive_avg_pool2d(tmp, (1,1))
            in_features = pooled.shape[1]

        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        cur = x
        for i, module in enumerate(self.backbone.features):
            cur = module(cur)
            key = str(i)
            if key in self.cbam_map:
                cur = self.cbam_map[key](cur)
        out = self.backbone.classifier(cur)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total

def test_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("RESULTADOS DE LA EVALUACIÓN FINAL")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")

    return avg_loss, accuracy, recall, f1, all_labels, all_preds

def plot_metrics(train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Val {metric_name}')
    plt.title(metric_name)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(labels, preds, class_names=['Benign', 'Malignant']):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix - Test')
    plt.show()


if __name__ == "__main__":
    base_dir = ""
    train_csv = "train.csv"
    val_csv = "val.csv"
    test_csv = "test.csv"

    batch_size = 8    
    img_size = (512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = MammogramDataset(train_csv, mode="train", base_dir=base_dir, img_size=img_size)
    val_dataset = MammogramDataset(val_csv, mode="val", base_dir=base_dir, img_size=img_size)
    test_dataset = MammogramDataset(test_csv, mode="test", base_dir=base_dir, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = ConvNeXtLarge_CBAM(num_classes=2, pretrained=True, img_size=img_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scaler = GradScaler()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    EPOCHS = 25
    best_val_loss = float("inf")
    best_model_path = "best_convnext_large_cbam.pth"

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train_loss: {train_loss:.4f} | Train_acc: {train_acc*100:.2f}%")
        print(f"  Val_loss:   {val_loss:.4f} | Val_acc:   {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")

        scheduler.step()

    plot_metrics(history['train_loss'], history['val_loss'], 'Loss')
    plot_metrics(history['train_acc'], history['val_acc'], 'Accuracy')

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    _, _, _, _, test_labels, test_preds = test_model(model, test_loader, criterion, device)
    plot_confusion_matrix(test_labels, test_preds)