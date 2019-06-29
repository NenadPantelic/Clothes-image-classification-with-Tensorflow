# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


print(dir(keras.datasets))
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_colorbar(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

#scale_plus_flag indicates if scale will be to scale up or scale down
def scale_image(images, scale_factor):
    return images * scale_factor


show_colorbar(train_images[0])
train_images = scale_image(train_images, 1/255.0)
test_images = scale_image(test_images, 1/255.0)

def plot_images_sample_grid(images, labels, grid_size, figsize):
    sample_size = len(images)

    plt.figure(figsize = figsize)
    for i in range(sample_size):
        plt.sublot(grid_size[0], grid_size(1), i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])

    plt.show()


images = train_images[:25]
labels = [class_names[train_labels[i]] for i in range(25)]
figsize = (10,10)
grid_size = (5,5)
plot_images_sample_grid(images, labels, grid_size, figsize )
