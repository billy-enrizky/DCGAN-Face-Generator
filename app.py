import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
from io import BytesIO

LATENT_SIZE = 128
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the generator model
# Define the generator architecture using nn.Sequential
generator = nn.Sequential(
    # First Transposed Convolutional Layer
    nn.ConvTranspose2d(LATENT_SIZE, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),         # Batch Normalization for stabilization
    nn.ReLU(True),               # ReLU activation function for non-linearity

    # Second Transposed Convolutional Layer
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    # Third Transposed Convolutional Layer
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    # Fourth Transposed Convolutional Layer
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    # Fifth Transposed Convolutional Layer (Output Layer)
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()                    # Tanh activation for output normalization
)

# Load the state_dict
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # Set the generator to evaluation mode

# Function to denormalize image tensors
def denorm(img_tensors):
    return img_tensors * MEAN[0] + STD[0]

# Function to generate anime faces using the loaded generator
def generate_faces(generator, num_images, randomness_level):
    generated_images = []

    for _ in range(num_images):
        # Generate random noise
        random_noise = torch.randn(1, LATENT_SIZE, 1, 1, device=device)
        random_noise *= randomness_level

        # Generate face using the loaded generator
        with torch.no_grad():
            # Use the generator function directly to generate the image
            generated_image = generator(random_noise)

        # Denormalize the generated image
        generated_image = denorm(generated_image)

        # Convert to NumPy array
        generated_image_np = (generated_image * 255).cpu().numpy().astype('uint8')

        # Convert to PIL Image and resize
        pil_image = Image.fromarray(generated_image_np.squeeze().transpose(1, 2, 0))
        resized_image = pil_image.resize((64, 64))
        generated_images.append(resized_image)

    return generated_images

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = f'generated-images-{index:0=4d}.png'

    # Save the generated images
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)

    # Optionally, display the saved images in a grid
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def main():
    st.image('DCGANfg.jpeg')
    st.title('DCGAN Project: Anime Face Generator')
    
    # Slider for selecting the randomness level
    randomness_level = st.slider("Randomness Level (The closer to 1, the clearer the images)", 
                                 0.0, 1.0, 0.5, step=0.1)
    
    # Button to generate faces
    if st.button("Generate Faces"):
        generated_images = generate_faces(generator, 64, randomness_level)
        st.write("Generated Anime Faces")
        
        # Display the generated images in a grid
        images_grid = make_grid([transforms.ToTensor()(img) for img in generated_images],
                                nrow=8).permute(1, 2, 0)
        
        # Convert the images_grid to PIL Image
        pil_image = Image.fromarray((images_grid.numpy() * 255).astype('uint8'))
        
        # Display the resized image
        st.image(pil_image, caption="Generated Anime Faces")

        # Download button for the generated image
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format='PNG')
        st.download_button(label='Download Generated Image', data=img_bytes.getvalue(),
                            file_name='generated_image.png', mime='image/png')


if __name__ == '__main__':
    main()