import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import caffe
import time

def vis_square(data,figname):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    plt.savefig(figname+".png")

if len(sys.argv) < 1:
    pritn("No path to image, use as follows: \n mnist_deply.py path/to/image.jpg")
    sys.exit()

model_path = "code/lab04/caffenet/deploy.prototxt"
weights_path = "code/lab04/caffenet/bvlc_reference_caffenet.caffemodel"

#Net loading parameters changed in Python 3
net = caffe.Net(model_path, 1, weights=weights_path)

# visualise one of the layers
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1), "conv1_params")

# load mean ImageNet image for subtraction
mu = np.load('code/lab04/caffenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_mean('data', mu) # set mean before we transform the image
transformer.set_channel_swap('data', (2,1,0)) # change from RGB to BGR

#  set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

#Loads the image then transform it
image = caffe.io.load_image(sys.argv[1]) #Loads the image
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image


#Use CPU for inferencing
caffe.set_mode_cpu()

cpuStartTime = time.time()
### perform classification
output = net.forward()
cpuEndTime = time.time()

print("Inferencing with CPU took {:.2f}ms".format((cpuEndTime-cpuStartTime)*1000.0))

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

#Load labels
labels = np.loadtxt('code/lab04/caffenet/synset_words.txt', str, delimiter='\t')
print('predicted class is:', labels[output_prob.argmax()])


#Find the index with the highest probablility
highest_index = -1
highest_probability = 0.0

for i in range(len(labels)):
    if output_prob[i] > highest_probability:
        highest_index = i
        highest_probability = output_prob[i]

#Print our result
if highest_index < 0:
    print "Did not detect a number!"
else:
    print "Digit {:s} detected with {:.2f}%  probability.".format(labels[highest_index], highest_probability*100.0)
