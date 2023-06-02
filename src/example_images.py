import os
from model import StyleTransfer

# Define the parent directories for content, style, and output images
content_parent = "deployment/assets/content_images/"
style_parent = "deployment/assets/style_images/"
output_parent = "deployment/assets/output_images/"

# Get a list of sorted content and style images
content_images = sorted(os.listdir(content_parent))
style_images = sorted(os.listdir(style_parent))

# Iterate through pairs of content and style images
for i, images in enumerate(zip(content_images, style_images)):
    content_path = content_parent + images[0]  # Get the path for the current content image
    style_path = style_parent + images[1]  # Get the path for the current style image
    output_path = output_parent + f"output_image{i+1}.jpg"  # Define the output path for the stylized image

    # Create a StyleTransfer object with specified parameters
    st = StyleTransfer(total_variation_weight=0, style_weight=1e-1, content_weight=1e4, epochs=20, steps_per_epoch=100, learning_rate=0.05)
    
    # Perform style transfer using the current content and style images, and save the result to the output path
    st.style_transfer(content_path, style_path, output_path)
