# Neural Style Transfer Project

This repository provides a TensorFlow-based implementation of Neural Style Transfer, along with a web interface built using Plotly Dash. Neural Style Transfer is a technique that allows you to apply the artistic style of one image to the content of another image, creating visually appealing and artistic results.

## Introduction
Neural Style Transfer combines the content of one image with the style of another image using deep neural networks. This implementation uses the VGG19 model pre-trained on the ImageNet dataset to extract content and style features from the input images. The content loss and style loss are then computed and used to optimize a target image that balances both content and style.

The web interface built with Plotly Dash provides a user-friendly way to apply Neural Style Transfer without the need for any coding. It allows users to upload their own content and style images, choose number of epochs, and visualize the generated stylized image.


![](assets\dashboard1.png)

![](assets\dashboard2.png)