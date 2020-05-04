'''
@author      : Anton 
@class       : MSDS 458
@date        : December 9, 2018
@description : A Python 3 script used to unzip and extract the flowers dataset
               into testing, training, and validation sets
@reference   : https://www.kaggle.com/anktplwl91/visualizing-what-your-convnet-learns/notebook
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                             IMPORTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import glob
import random
import shutil

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                             GLOBAL VARIABLES
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

VAL_SIZE = 0.15
TEST_SIZE = 0.2

train_images = []
val_images = []
test_images = []

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

orig_path = 'flowers'
train_path = orig_path + '/train'
val_path = orig_path + "/validation"
test_path = orig_path + "/test"

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                MAIN
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('[+] Running...')
print(' > preparing data')

# Unzip the dataset. Note: This works for Google Colab. If not using Colab, 
# commment out this section and manually 'unzip flowers.zip'
#if not os.path.isdir(orig_path):
#    print(' > unzipping flowers.zip')
#    !unzip flowers.zip
#else:
#    print(' > directory exists: {}'.format(orig_path))

# Create sub-directories for train, val, and test images. Note: os commands 
# are geared toward *nix systems
if not os.path.isdir(train_path):
    print(' > making sub-directory: {}'.format(train_path))
    os.mkdir(train_path, 0o777);
else:
    print(' > sub-directory exists: {}'.format(train_path))
if not os.path.isdir(val_path):
    print(' > making sub-directory: {}'.format(val_path))
    os.mkdir(val_path, 0o777);
else:
    print(' > sub-directory exists: {}'.format(val_path))    
if not os.path.isdir(test_path):
    print(' > making sub-directory: {}'.format(test_path))
    os.mkdir(test_path, 0o777);
else:
    print(' > sub-directory exists: {}'.format(test_path))

train_exists = False
val_exists = False
test_exists = False

# Check if the directories exists
if len(os.listdir(train_path) ) == 0:
    train_exists = False
else:    
    train_exists = True 
if len(os.listdir(val_path) ) == 0:
    val_exists = False
else:    
    val_exists = True
if len(os.listdir(test_path) ) == 0:
    test_exists = False
else:    
    test_exists = True

f = []
dirs = []
# If the directories don't exists, create them and move images
# to the appropriate directories based on the 0.2,0.15 split
if not train_exists and not val_exists and not test_exists:
    print(' > creating train, test, and val data')
    for c in classes:
        
        # Get a list of jpg file names
        img_list = glob.glob(orig_path + '/' + c + '/*.jpg')

        # Take a random sample from the images pool
        val_images = random.sample(img_list, int(VAL_SIZE * len(img_list)))
        img_list = [f for f in img_list if f not in val_images]
        test_images = random.sample(img_list, int(TEST_SIZE * len(img_list)))
        train_images = [f for f in img_list if f not in test_images]

        # Create train, test, and validationd directories
        os.mkdir(train_path + "/" + str(c), 0o777);
        os.mkdir(val_path + "/" + str(c), 0o777);
        os.mkdir(test_path + "/" + str(c), 0o777);

        # Copy the sampled images to the appropriate directory
        for f in train_images:
            shutil.copy(f, train_path + "/" + str(c))

        for f in val_images:
            shutil.copy(f, val_path + "/" + str(c))

        for f in test_images:
            shutil.copy(f, test_path + "/" + str(c))
else:
    print(' > train, test, and val data exists')
        
print(' > data export and organization complete')

'''
@author      : Anton 
@class       : MSDS 458
@date        : December 9, 2018
@description : A Python 3 script used analyze the intermediate layers (hidden layers) of
               of a test image. Analayze activations, filters, and CAM
@reference   : https://www.kaggle.com/anktplwl91/visualizing-what-your-convnet-learns/notebook
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                             IMPORTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from time import time
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint

get_ipython().run_line_magic('matplotlib', 'inline')

"""Function to enable GPU and clears GPU memory"""
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                             GLOBAL VARIABLES
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

BASE_PATH = 'flowers'
TRAIN_DIR = BASE_PATH + '/train'
VALID_DIR = BASE_PATH + '/validation'
IMG_SIZE = (299, 299, 3)
BATCH_SIZE = 32

limit_mem()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                MAIN
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Check for the existance of a model. If model exists, begin analysis of layers. If 
# not exist, train a model
if not os.path.isfile('inceptionv3.h5'):

    print('\n[+] Data generator starting')
    
    # Creating train and validation data generator using Keras' ImageDataGenerator module
    train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, 
                                       height_shift_range=0.2, horizontal_flip=True, 
                                       rescale=1./255)
    
    # Rescale the RGB pixel values to between 0 and 1
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Read the images from the directories into a numpy array for both train and val 
    train_gen = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                  target_size=(IMG_SIZE[0], IMG_SIZE[1]), 
                                                  batch_size=BATCH_SIZE, class_mode="categorical")

    val_gen = val_datagen.flow_from_directory(VALID_DIR, 
                                              target_size=(IMG_SIZE[0], IMG_SIZE[1]), 
                                              batch_size=BATCH_SIZE, class_mode="categorical")

    print(' > train and val data generation complete')
    print('\n[+] Creating inception model and training')

    # Creating InceptionV3 model and training it
    inp = Input(IMG_SIZE)
    inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, 
                            input_shape=IMG_SIZE, pooling='avg')
    x = inception.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(5, activation='softmax')(x)

    complete_model = Model(inp, out)

    # Compile the model
    complete_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print (complete_model.summary())

    # Implement callbacks (early stopping and a checkpoint of the best weights)
    callbacks = [EarlyStopping(monitor='loss', patience=2), 
                 ModelCheckpoint(filepath='best_weight.hdf5', 
                                 monitor='loss', save_best_only=True)]

    # Begin the training process. Recommended AWS P2/P3 instance or Google Colab
    history = complete_model.fit_generator(train_gen, 
                                           steps_per_epoch=92,
                                           validation_steps=10,
                                           epochs=10, 
                                           validation_data=val_gen, 
                                           callbacks=callbacks,
                                           verbose=1)

    # summarize history for accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history['acc'], label='Train')
    ax.plot(history.history['val_acc'], label='Test')
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    leg = ax.legend()
    plt.savefig('accuracy.jpg')
    
    #  summarize history for loss
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    leg = ax.legend()
    plt.savefig('loss.jpg')

    # Save the history as a pickle
    print (' > saving model training history')
    with open('inceptionv3_histobject.pkl', 'wb') as fi:
        pickle.dump(history.history, fi)

    # Save the complete model to disk
    complete_model.save('inceptionv3.h5')
    print(' > model saved to: inceptionv3.h5')
else:
    # If the model exists, load it
    print(' > found model: inception3.h5. Loading...')
    complete_model = load_model('inceptionv3.h5')
    print(' > loaded saved model: inceptionv3.h5')

# Getting outputs for intermediate convolution layers by running prediction on test image
print('\n[+] Getting outputs for intermediate conv layers on test image')

# Get layer outputs from the model
layer_outputs = [layer.output for layer in complete_model.layers[:50]]
layer_outputs = layer_outputs[1:]
test_image = 'flowers/test/daisy/34665595995_13f76d5b60_n.jpg'

# Load the test image, expand dims, and change RGB values
img = image.load_img(test_image, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Create a model (load the weights). Predict the output of the model.
activation_model = Model(inputs=complete_model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

# Define the intermediate (hidden) layers that will be analyzed
layer_names = ['conv2d_1', 'activation_1', 'conv2d_4', 'activation_4', 'conv2d_9', 'activation_9']
activ_list = [activations[1], activations[3], activations[11], activations[13], activations[18], activations[20]]

images_per_row = 16

# For each layer being analyzed, create a visualization of the outputs
for layer_name, layer_activation in zip(layer_names, activ_list):
    print(' > layer: {}'.format(layer_name))
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='plasma')
    plt.savefig(layer_name+"_grid.jpg", bbox_inches='tight')

'''Function for displaying filters as images'''
def deprocess_image(x):
    
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

''' Function for generating patterns for given layer starting grom empty input image
    and then applying SGD for maximizing the response of a particular filter in a layer'''
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = activation_model.get_layer(layer_name).output
    #layer_output[1:]
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, complete_model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([complete_model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)
  
# Generating convolution layer filters for intermediate layers using above utility functions
print('\n[+] Generating layer filters')

layer_name = 'conv2d_4'
size = 299
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

# Extract and genreate a visualization of the filters used in the 4th conv2d layer
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img    

plt.figure(figsize=(20, 20))
plt.title('Convolution Layer Filters for Intermediate Layers')
plt.grid(False)
plt.imshow((results * 255).astype(np.uint8), aspect='auto')
plt.savefig('filter.jpg')

# Initialize an InceptionV3 model and make predictions on a test image. Create a 
# CAM of the last layer of the model, which is a mixed layer. Superimpose the CAM
# over the origninal image.
print('\n[+] Make predictions using InceptionV3 model')

# Create the InceptionV3 model
model = InceptionV3(weights='imagenet')
img_path = 'flowers/test/daisy/34665595995_13f76d5b60_n.jpg'

# Load the test image, expand dims, and preprocess input
img = image.load_img(img_path, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict the output and print to stdout
preds = model.predict(x)
print(' > image: {}'.format(img_path))
print(' > predicted: {}'.format(decode_predictions(preds, top=3)[0]))

# 985 is the class index for class 'Daisy' in Imagenet dataset on which the model is pre-trained
flower_output = model.output[:, 985]
last_conv_layer = model.get_layer('mixed10')

grads = K.gradients(flower_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

# 2048 is the number of filters/channels in 'mixed10' layer
for i in range(2048):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# Generate the heatmap
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.savefig('heatmap.jpg')

# Using cv2 to superimpose the heatmap on original image to clearly illustrate 
# activated portion of image
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('superimposed_img.jpg', superimposed_img)
