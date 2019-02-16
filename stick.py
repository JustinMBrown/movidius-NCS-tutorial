import numpy as np
from mvnc import mvncapi
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()

# Get a list of available device identifiers
device_list = mvncapi.enumerate_devices()
# Initialize a Device
device = mvncapi.Device(device_list[0])
# Initialize the device and open communication
device.open()

# Reading the file from disk
GRAPH_FILEPATH = './graph'
with open(GRAPH_FILEPATH, mode='rb') as f:
    graph_buffer = f.read()

# Initialize a Graph object
graph = mvncapi.Graph('graph1')

# Initialize the input/output queue objects
input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_buffer)

# 28x28 image of the number "8", uint8 datatype
tensor = x_train[image_index]

# We need to convert it to float32, that's the only thing the Movidius stick accepts
tensor = tensor.astype(np.float32)

# The ranges are from 0 to 255, but we need our ranges to be between 0 and 1
tensor /= 255

# You input into this object
graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, tensor, 'user object')

# And you read from this object
output, user_obj = output_fifo.read_elem()

# Select the highest prediction, and get it's index.
# Remeber, the index represents the actual number
prediction_index = np.argmax(output)
print(prediction_index)


#boiler plate
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()

