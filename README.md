# DCGAN: Anime Face Generator

This project involves the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate anime faces. The project consists of two main components: the model training notebook (`model.ipynb`) and the web application using Streamlit (`app.py`).

## Model Training (`model.ipynb`)

### Importing Necessary Libraries

```python
import os, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
```

### Dataset Description

The Anime Face Dataset comprises 21,551 anime faces sourced from www.getchu.com. All images in the dataset should be manually resized to a standard size of 64 x 64 pixels to ensure uniformity.

### Data Preprocessing and Loading

```python
# CONSTANTS
IMAGE_SIZE = 64
BATCH_SIZE = 128
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
DATA_DIR = './data/'

# Data transformation and loading
train_ds = ImageFolder(DATA_DIR, transform=tt.Compose([tt.Resize(IMAGE_SIZE),
                                                       tt.CenterCrop(IMAGE_SIZE),
                                                       tt.ToTensor(),
                                                       tt.Normalize(mean=MEAN, std=STD)]))
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# Utility functions for image display
def denorm(img_tensors):
    return img_tensors * MEAN[0] + STD[0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break
```

### Transferring Data to the GPU Device

```python
# GPU device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dl = DeviceDataLoader(train_dl, device)
```

### Discriminator Neural Network

The discriminator architecture is defined using `nn.Sequential`.

```python
discriminator = nn.Sequential(
    # Convolutional layers...
)

# Move the discriminator to the specified device and display the summary
discriminator = to_device(discriminator, device)
summary(discriminator, (3, 64, 64))
```

### Generator Neural Network

The generator architecture is defined using `nn.Sequential`.

```python
generator = nn.Sequential(
    # Transposed Convolutional layers...
)

# Move the generator to the specified device and display the summary
generator = to_device(generator, device)
summary(generator, input_size=(LATENT_SIZE, 1, 1))
```

### Initiating to Save Generated Images

```python
# Generate random latent vectors and fake images
xb = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1)
fake_images = generator(xb)

# Display the generated fake images
show_images(fake_images)

# Create a directory to save generated images
sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)
```

### Training the DCGAN

```python
# Define functions for training the GAN model
def fit(model, criterion, epochs, lr, start_idx=1):
    # Training code...

# Define the GAN model and training parameters
model = {'discriminator': discriminator.to(device), 'generator': generator.to(device)}
criterion = {'discriminator': nn.BCELoss(), 'generator': nn.BCELoss()}
lr = 0.0002
epochs = 50

# Train the GAN model
history = fit(model, criterion, epochs, lr)
```

### Visualizing Losses

```python
# Plot discriminator and generator losses
losses_g, losses_d, _, _ = history
plt.figure(figsize=(15, 6))
plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('The Discriminator Loss and the Generator Loss');
```

### Showing Generated Images

```python
# Display the last generated image
generated_img = cv2.imread(f'./generated/generated-images-00{epochs}.png')
generated_img = generated_img[:, :, [2, 1, 0]]
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xticks([]); ax.set_yticks([])
ax.imshow(generated_img);
```

### Saving the Generator Model as a .pkl File

```python
# Save the generator model as a .pkl file
generator_model = generator
with open('generator_model.pkl', 'wb') as f:
    pickle.dump(generator_model, f)
```

## Streamlit User-Friendly Version

A user-friendly version of the DCGAN Anime Face Generator is implemented using Streamlit in the `app.py` file. Users can generate anime faces with a single click.

```python
# Importing necessary libraries in app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Load the generator model
generator = nn.Sequential(
    # Transposed Convolutional layers...
)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # Set the generator to evaluation mode

# Function to denormalize image tensors
def denorm(img_tensors):
    return img_tensors * MEAN[0] + STD[0]

# Function to generate anime faces using the loaded generator
def generate_faces(generator, num_images, randomness_level):
    # Generation code...
    return generated_images

# Streamlit app functions...
```

The Streamlit app includes a slider for selecting the randomness level, a button to generate faces, and an option to download the generated image.

To explore the Streamlit version, click the button below:

[Generate Anime Faces](https://dcgan-face-generator.streamlit.app/)

## Deployment

To deploy the web application, run the following command in the terminal:

```bash
streamlit run app.py
```
