# Forest Persistence Segmentation Using U-Net

## Overview

This project performs **forest persistence segmentation** using a **U-Net** deep learning model based on satellite images exported from **Google Earth Engine (GEE)**. We have created a account in the gee- google earth engine account which would require a google cloud account where its registered and a unique id is generated (erudite-imprint-458001-e5) for the project , this id is required to access the data in the google earth engine website and then the data is exported from it to see the forest persistence segmentation. The forest persistence segmentation data is from ee.Image("projects/forestdatapartnership/assets/community_forests/ForestPersistence_2020") . The website mentioned i.e ForestPersistence_2020 image shows how stable forests were over 20 years, and U-Net learns to predict forest areas pixel-by-pixel from it. We used: - Earth Engine for data access, - PyTorch for model training, - OpenCV for image preprocessing, -Matplotlib for visualization. We have used colab for running this code and used google earth engine for the datset

## Step-by-Step Execution

| Step | Description                                   |
|:-----|:----------------------------------------------|
| 1    | Download satellite data (Earth Engine export) |
| 2    | Preprocess TIFF images into PNG format        |
| 3    | Build U-Net segmentation model                |
| 4    | Prepare custom PyTorch dataset                |
| 5    | Train U-Net model                             |
| 6    | Predict mask on whole image                   |
| 7    | Evaluate metrics (IoU, Precision, Recall)     |
| 8    | Visualize input, prediction, and metrics      |

------------------------------------------------------------------------

## Step 1: Mount Google Drive

``` python
from google.colab import drive
drive.mount('/content/drive')
```

------------------------------------------------------------------------

## Step 2: Google Earth Engine Setup and Export

``` python
import ee
import geemap

ee.Authenticate()
ee.Initialize(project='erudite-imprint-458001-e5')

roi = ee.Geometry.BBox(-60.0, -10.0, -59.5, -9.5)
gfc = ee.Image('UMD/hansen/global_forest_change_2020_v1_8')
loss = gfc.select('loss')

task = ee.batch.Export.image.toDrive(
    image=loss.clip(roi),
    description='forest_loss_amazon',
    folder='EarthEngineExports',
    fileNamePrefix='forest_loss_amazon',
    region=roi,
    scale=30,
    crs='EPSG:4326',
    maxPixels=1e9
)
task.start()

print("Export started from Earth Engine. Wait for download from Google Drive.")
```

------------------------------------------------------------------------

## Step 3: Convert .tif to .png

``` python
from PIL import Image
import os

tif_path = "/content/drive/MyDrive/mask/persistent_forest_colored_washington.tif"
image = Image.open(tif_path)

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

image.convert("RGB").save("data/images/forest_persistence_raw.png")
image.convert("L").save("data/masks/forest_persistence_raw.png")

print("Image and mask saved for training.")
```

------------------------------------------------------------------------

## Step 4: Define U-Net Model

``` python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        return torch.sigmoid(self.final(dec1))
```

------------------------------------------------------------------------

## Step 5: Create Dataset Loader

``` python
from torch.utils.data import Dataset, DataLoader
import cv2

class ForestDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, mask
```

Helper functions for padding and cropping:

``` python
import torch.nn.functional as F

def pad_to_multiple(x, multiple=32):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, h, w

def unpad(x, h, w):
    return x[:, :, :h, :w]
```

------------------------------------------------------------------------

## Step 6: Train U-Net

``` python
import torch.optim as optim
import matplotlib.pyplot as plt

train_dataset = ForestDataset("data/images", "data/masks")

if len(train_dataset) == 0:
    print("Add image and mask files in 'data/images' and 'data/masks' before training.")
else:
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        for img, mask in train_loader:
            img, orig_h, orig_w = pad_to_multiple(img)
            mask, _, _ = pad_to_multiple(mask)

            pred = model(img)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.savefig("outputs/loss_curve.png")
    plt.show()
```

------------------------------------------------------------------------

## Step 7: Predict on Full Image

``` python
import os
model.eval()

input_path = "data/images/forest_persistence_raw.png"
image = cv2.imread(input_path)

image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
image_padded, orig_h, orig_w = pad_to_multiple(image_tensor)

with torch.no_grad():
    pred_padded = model(image_padded)

prediction = unpad(pred_padded, orig_h, orig_w)

pred_mask = prediction.squeeze().numpy() * 255
cv2.imwrite("outputs/predicted_whole_image.png", pred_mask.astype('uint8'))
```

------------------------------------------------------------------------

## Step 8: Calculate Metrics (IoU, Precision, Recall)

``` python
import numpy as np

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    pred_binary = (pred_mask >= threshold).astype(np.uint8)
    true_binary = (true_mask >= threshold).astype(np.uint8)

    intersection = (pred_binary & true_binary).sum()
    union = (pred_binary | true_binary).sum()
    iou = intersection / (union + 1e-6)

    tp = (pred_binary * true_binary).sum()
    fp = (pred_binary * (1 - true_binary)).sum()
    fn = ((1 - pred_binary) * true_binary).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return iou, precision, recall
```

------------------------------------------------------------------------

## Step 9: Visualization

``` python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

img_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
pred_np = prediction.squeeze().numpy()

colors = [
    (1.0, 1.0, 1.0), (1.0, 1.0, 0.0),
    (0.5647, 0.9333, 0.5647), (0.0, 0.5, 0.0), (0.0, 0.3922, 0.0)
]
forest_cmap = LinearSegmentedColormap.from_list("forest_palette", colors, N=256)

img_gray = img_np.mean(axis=2)

fig, axs = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1, 0.05]})

im0 = axs[0].imshow(img_gray, cmap=forest_cmap, vmin=0, vmax=1)
axs[0].set_title("Input Image with Forest Palette")
axs[0].axis("off")

axs[1].imshow(pred_np, cmap='gray')
axs[1].set_title("Predicted Mask (Grayscale)")
axs[1].axis("off")

cbar = plt.colorbar(im0, cax=axs[2])
cbar.set_label('Forest Persistence', rotation=270, labelpad=15)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])

plt.tight_layout()
plt.show()
```

------------------------------------------------------------------------

## Sample Datasets taken for this project

-   [Dataset 1](https://code.earthengine.google.com/23425646a6731bcc50026fd7e49a7285?noload=1)
-   [Dataset 2](https://code.earthengine.google.com/02939ee78eb9c1a879b7dffc204daa4f?noload=1)
-   [Dataset 3](https://code.earthengine.google.com/c135d03b3b7671c21cefead3b18a7f56?noload=1)

------------------------------------------------------------------------
