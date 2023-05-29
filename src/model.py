import tensorflow as tf
import IPython.display as display
import numpy as np
import PIL.Image

# Define the StyleContentModel class
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)  # Create a VGG model with specified style and content layers
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0  # Scale the input to the range [0, 255]
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)  # Preprocess the input for VGG19
        outputs = self.vgg(preprocessed_input)  # Pass the preprocessed input through the VGG model
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])  # Split the outputs into style and content

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]  # Compute the Gram matrix for each style output

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}  # Create a dictionary mapping content layer names to their outputs

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}  # Create a dictionary mapping style layer names to their outputs

        return {'content': content_dict, 'style': style_dict}

# Convert a tensor to an image
def tensor_to_image(tensor):
    tensor = tensor*255  # Scale the tensor to the range [0, 255]
    tensor = np.array(tensor, dtype=np.uint8)  # Convert the tensor to a NumPy array of unsigned 8-bit integers
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)  # Create a PIL image from the NumPy array

# Load an image from a file path
def load_img(path_to_img, max_dim):
    img = tf.io.read_file(path_to_img)  # Read the image file
    img = tf.image.decode_jpeg(img, channels=3)  # Decode the JPEG image
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert the image to float32 format

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)  # Get the shape of the image
    long_dim = max(shape)  # Find the longer dimension of the image
    scale = max_dim / long_dim  # Calculate the scale factor

    new_shape = tf.cast(shape * scale, tf.int32)  # Calculate the new shape of the image after scaling

    img = tf.image.resize(img, new_shape)  # Resize the image
    img = img[tf.newaxis, :]  # Add a batch dimension
    return img
    
# Create a VGG model that returns a list of intermediate output values
def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')  # Load the VGG19 model
    vgg.trainable = False  # Set the VGG19 model as non-trainable

    outputs = [vgg.get_layer(name).output for name in layer_names]  # Get the outputs of specified layers

    model = tf.keras.Model([vgg.input], outputs)  # Create a new model that takes VGG19 input and outputs the specified layer outputs
    return model

# Compute the Gram matrix of an input tensor
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)  # Compute the pairwise dot products of the input tensor
    input_shape = tf.shape(input_tensor)  # Get the shape of the input tensor
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)  # Calculate the number of locations in the input tensor
    return result/(num_locations)  # Normalize the result by the number of locations

# Clip the pixel values of an image tensor to the range [0, 1]
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Class for performing style transfer
class StyleTransfer:
    def __init__(self, total_variation_weight=0, style_weight=1e-2,
                 content_weight=1e4, steps_per_epoch=100,
                 learning_rate=0.02, max_dim = 1200):
        self.total_variation_weight = total_variation_weight
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.max_dim = max_dim

    def style_transfer(self, content_path, style_path, output_path, epochs = 20):
        self.epochs = epochs
        content_image = load_img(content_path, self.max_dim)  # Load the content image
        style_image = load_img(style_path, self.max_dim)  # Load the style image

        content_layers = ['block5_conv2']  # Specify the content layers for the StyleContentModel

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']  # Specify the style layers for the StyleContentModel

        style_extractor = vgg_layers(style_layers)  # Create a VGG model for style extraction
        style_outputs = style_extractor(style_image*255)  # Extract style features from the style image

        num_content_layers = len(content_layers)  # Get the number of content layers
        num_style_layers = len(style_layers)  # Get the number of style layers

        extractor = StyleContentModel(style_layers, content_layers)  # Create a StyleContentModel

        style_targets = extractor(style_image)['style']  # Get the style targets from the style image
        content_targets = extractor(content_image)['content']  # Get the content targets from the content image

        image = tf.Variable(content_image)  # Initialize the image variable for optimization

        opt = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.99, epsilon=1e-1)  # Create an Adam optimizer

        step = 0  # Initialize the step counter
        for n in range(self.epochs):
            for m in range(self.steps_per_epoch):
                step += 1
                with tf.GradientTape() as tape:
                    outputs = extractor(image)  # Pass the image through the StyleContentModel
                    style_outputs = outputs['style']
                    content_outputs = outputs['content']
                    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                                           for name in style_outputs.keys()])  # Compute the style loss
                    style_loss *= self.style_weight / num_style_layers

                    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                                             for name in content_outputs.keys()])  # Compute the content loss
                    content_loss *= self.content_weight / num_content_layers
                    loss = style_loss + content_loss  # Compute the total loss
                    loss += self.total_variation_weight*tf.image.total_variation(image)  # Add the total variation loss
                grad = tape.gradient(loss, image)  # Compute the gradients of the loss with respect to the image
                opt.apply_gradients([(grad, image)])  # Apply the gradients to update the image
                image.assign(clip_0_1(image))  # Clip the pixel values of the image to the range [0, 1]
                print(".", end='', flush=True)
            display.clear_output(wait=True)
            print("Train step: {}".format(step))  # Print the current step

        tensor_to_image(image).save(output_path)  # Save the final image