import sys
import os
import numpy as np
import caffe
import time

model_path = "mnist_lenet_deploy.prototxt"
weights_path = "dl_lenet_snapshot__iter_10000.caffemodel"

caffe.set_mode_cpu() # Set as CPU mode

net = caffe.Net(model_path, 1, weights=weights_path)

#image = caffe.io.load_image(sys.argv[1], False) #Loads the image from the first argument variable

imagefilenames = ['data/mnist_three.png', 'data/mnist_six.png', 'data/mnist_nine.png']
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

# reshape data to allow for 3 images
net.blobs['data'].reshape(3,        # batch size
                          1,         # 3-channel (BGR) images
                          28, 28)  # image size is 227x227

print 'Loading images'
i=0
for imagefilename in imagefilenames:
    image = caffe.io.load_image(imagefilename, False)
    transformed_image = transformer.preprocess('data', image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[i,...] = transformed_image
    i = i+1

### perform classification
print 'starting classification'
startTime = time.time()
output = net.forward()
endTime = time.time()
print("Inferencing with CPU took {:.2f}ms".format((endTime-startTime)*1000.0))

startTime = time.time()
caffe.set_mode_gpu()
caffe.set_device(0)
net.forward()
endTime = time.time()
print("Inferencing with GPU took {:.2f}ms".format((endTime-startTime)*1000.0))

# Text representation of the digit
digits_label = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

# loop over all files & find the result
for i in range(0, len(imagefilenames)):
    print 'For file ', imagefilenames[i]
    output_prob = output['loss'][i]  # the output probability vector for the first image in the batch

    #Find the index with the highest probablility
    highest_index = -1
    highest_probability = 0.1 #If nothing's more than 10% sure then don't print out anything
    
    for i in range(0,9):
        if output_prob[i] > highest_probability:
            highest_index = i
            highest_probability = output_prob[i]

    #Print our result
    if highest_index < 0:
        print "Did not detect a number!"
    else:
        print "Digit "+ str(digits_label[highest_index]) + " detected with " + str(highest_probability*100.0)+"%  probability."


