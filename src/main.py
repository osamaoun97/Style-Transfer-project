import os
from model import StyleTransfer

conent_parent = "assets/content_images/"
style_parent = "assets/style_images/"
output_parent = "assets/output_images/"

content_images = sorted(os.listdir(conent_parent))
style_images = sorted(os.listdir(style_parent))

for i, images in enumerate(zip(content_images, style_images)):
    content_path = conent_parent + images[0]
    style_path = style_parent + images[1]
    output_path = output_parent + f"output_image{i+1}.jpg"

    st = StyleTransfer(total_variation_weight= 0,style_weight=1e-1, content_weight=1e4, epochs = 20, steps_per_epoch = 100, learning_rate=0.05)
    st.style_transfer(content_path, style_path, output_path)