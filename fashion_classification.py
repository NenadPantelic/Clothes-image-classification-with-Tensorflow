# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#utils libs and modules
import numpy as np
from matplotlib import pyplot as plt
from data_process_and_visualization import *


#load dataset - 60k images for training and 10k for test
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


for i in range(10):
    print(np.count_nonzero(train_labels == i))

#region DATASET EXPLORE
print_dataset_info(train_images, train_labels)
print_dataset_info(test_images, test_labels)






show_colorbar(train_images[0])
train_images = scale_image(train_images, 1/255.0)
test_images = scale_image(test_images, 1/255.0)

#show_colorbar(train_images[0])



images = train_images[:25]
labels = [class_names[train_labels[i]] for i in range(25)]
figsize = (10,10)
grid_size = (5,5)
plot_images_sample_grid(images, labels, grid_size, figsize )


#region MODEL BUILDING

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=25)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy = {:.2f}, test loss = {:.2f}'.format(test_acc, test_loss))


predictions = model.predict(test_images)



#region PREDICTION AND TESTING

#test for 0th image sample
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels, True)
plt.show()

#test for 15th image sample

i = 15
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels, True)
plt.show()

#test for 12th image sample

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels, True)
plt.show()

#test for 100th image sample

i = 100
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels, True)
plt.show()

#blue color - match, red color - nonmatch
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()



img = test_images[0]

print(img.shape)



img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)
plt.figure(figsize=(1, 2))

plt.subplot(1, 2, 1)
plot_image(0, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))
