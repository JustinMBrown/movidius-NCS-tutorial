import numpy as np
from mvnc import mvncapi
import matplotlib as plt
import tensorflow as tf
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

# With your webcam plugged in, this will fetch a reference to it
cam = cv2.VideoCapture(0)
# Set the capture size to 28x28
# Note, this will only force the capture to shrink up untill a certain point.
# For me, this was around 150x120. But it's better than nothing
cam.set(3,28)
cam.set(4,28)

while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()

    # Convert the fame to gray-scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Do the final resizing to 28x28
    resize = cv2.resize(gray, (28, 28))
    # Convert to float32, just like before
    tensor = resize.astype(np.float32)
    # Force the values to be between 0 and 1
    tensor /= 255
    # Display the resulting frame
    #cv2.imshow('frame', tensor)

    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, tensor, 'user object')

    output, user_obj = output_fifo.read_elem()

    # Print the prediction just like before
    print(np.argmax(output))

    # Press q to kill. Or just hit the stop button on Pycharm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break


#boiler plate
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()

