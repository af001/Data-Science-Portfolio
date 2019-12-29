# base Python packages
import os
import pickle
import time
import numpy as np
import tensorflow as tf
import gzip
import binascii
import struct
import matplotlib.pyplot as plt
from six.moves.urllib.request import urlretrieve
%matplotlib inline  

# ensure common functions across Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

WORK_DIRECTORY = r'C:\Users\Anton\Desktop'
PICKLE_FILE = (r'%s\mnist_data.pickle' % WORK_DIRECTORY)
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
RANDOM_SEED = 42
NUM_RUNS = 1

def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # Max pooling. The kernel size spec ksize also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=RANDOM_SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = np.zeros([10, 10], np.float32)
    bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    
    return error, confusions

def get_predictions(predictions, labels):
    return tf.equal(np.argmax(predictions, 1), np.argmax(labels, 1))

def accuracy(correct_prediction):
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# read data from pickle file
print('[+] Reading data from pickle file:')
with open('mnist_data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

# extract objects from the dictionary object data
train_data = data['train_data']
train_labels = data['train_labels'] 
validation_data = data['validation_data'] 
validation_labels = data['validation_labels'] 
test_data = data['test_data'] 
test_labels = data['test_labels']  

train_size = train_labels.shape[0]
    
# check data from pickle load
print('train_data object:', type(train_data), train_data.shape)    
print('train_labels object:', type(train_labels),  train_labels.shape)  
print('validation_data object:', type(validation_data),  validation_data.shape)  
print('validation_labels object:', type(validation_labels),  validation_labels.shape)  
print('test_data object:', type(test_data),  test_data.shape)  
print('test_labels object:', type(test_labels),  test_labels.shape)  

print('\nRead from pickle file complete...')

# shuffle the data
train_shuffle = np.c_[train_data.reshape(len(train_data), -1), train_labels.reshape(len(train_labels), -1)]
train_data = train_shuffle[:, :train_data.size//len(train_data)].reshape(train_data.shape)
train_labels = train_shuffle[:, train_data.size//len(train_data):].reshape(train_labels.shape)

###Define the model

# create a symbolic variable --> set train_data_node/train_labels_data 
# as a placeholders
train_data_node = tf.placeholder(
  tf.float32,
  shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, NUM_LABELS))

# For the validation and test data, we'll just hold the entire dataset in
# one constant node.
validation_data_node = tf.constant(validation_data)
train_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, NUM_LABELS))

# The variables below hold all the trainable weights. For each, the
# parameter defines how the variables will be initialized.
# These are the convolution layers.
conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                      stddev=0.1,
                      seed=RANDOM_SEED))
conv1_biases = tf.Variable(tf.zeros([32]))
conv2_weights = tf.Variable(
  tf.truncated_normal([5, 5, 32, 64],
                      stddev=0.1,
                      seed=RANDOM_SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
fc1_weights = tf.Variable(  # fully connected, depth 512.
  tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                      stddev=0.1,
                      seed=RANDOM_SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
  tf.truncated_normal([512, NUM_LABELS],
                      stddev=0.1,
                      seed=RANDOM_SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

print('\n[+] Done feeding training samples and labels into the graph!!')

# stamp multiple copies of the graph for training, testing, and validation
# Training computation: logits + cross-entropy loss. 
logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels=train_labels_node, logits=logits))

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0)
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                       0.9).minimize(loss,
                                                     global_step=batch)

# Predictions for the minibatch, validation set and test set.
train_prediction = tf.nn.softmax(logits)
# We'll compute them only once in a while by calling their {eval()} method.
validation_prediction = tf.nn.softmax(model(validation_data_node))

test_data_node = tf.placeholder(tf.float32,shape=(100, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
test_labels_node = tf.placeholder(tf.float32,shape=(100, NUM_LABELS))
test_size = train_labels.shape[0]

logits = model(test_data_node, True)
test_prediction = tf.nn.softmax(logits)

print('[+] Done building training, test, and validation graphs!')

# go through training loop and periodically evaluate loss and error
# Create a new interactive session that we'll use in
# subsequent code cells.
s = tf.InteractiveSession()

# Use our newly created session as the default for 
# subsequent operations.
s.as_default()

# Initialize all the variables we defined above.
tf.global_variables_initializer().run()

# perform operations on the graph. Do one round of training to begin.

# Grab the first BATCH_SIZE examples and labels.
batch_data = train_data[:BATCH_SIZE, :, :, :]
batch_labels = train_labels[:BATCH_SIZE]

# This dictionary maps the batch data (as a numpy array) to the
# node in the graph it should be fed to.
feed_dict = {train_data_node: batch_data,
             train_labels_node: batch_labels}

# Run the graph and fetch some of the nodes.
_, l, lr, predictions = s.run(
  [optimizer, loss, learning_rate, train_prediction],
  feed_dict=feed_dict)

print('\n[+] Estimation for one round of training:')
print('Predictions: \n', predictions[0])

# But, predictions is actually a list of BATCH_SIZE probability vectors.
print('Prediction shape:', predictions.shape)

# The highest probability in the first entry.
print('First prediction', np.argmax(predictions[0]))

# So, we'll take the highest probability for each vector.
print('All predictions', np.argmax(predictions, 1))
print('Batch labels', np.argmax(batch_labels, 1))

print('\n[+] Perform batch training using %s rounds and batch size of %s' % (NUM_RUNS, BATCH_SIZE))

run_times = {}
step_times = {}
step_count=0
#### predicted and label classes --> error rate
for run in range(NUM_RUNS):
    # start the timer
    start_time = time.clock()
    print('\n[+] Run number: %s' % str(run+1))
    steps = train_size // BATCH_SIZE
    for step in range(steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        
        # start the timer
        start_time_step = time.clock()
    
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = s.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
        
        # end the run timer
        end_time_step = time.clock()
        step_times[step_count] = (end_time_step - start_time_step)*1000
        step_count+=1
        
        # Print out the loss periodically.
        if step % 100 == 0:
            error, _ = error_rate(predictions, batch_labels)
            print('[*] Step %d of %d' % (step, steps))
            print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
            print('Validation error: %.1f%%' % error_rate(
                  validation_prediction.eval(), validation_labels)[0])
            _predictions = get_predictions(predictions, batch_labels)
            print("Accuracy: %.2f%%" % s.run(accuracy(_predictions), feed_dict=feed_dict))

        # end the run timer
        end_time = time.clock()
        run_times[run] = (end_time - start_time)*1000
    
print('\n[+] Execution Time:')
print('Total Runs: %s' % NUM_RUNS)
print('Average Step Time: %s' % (sum(step_times.values())/len(step_times)))

total_time = 0
for run in range(NUM_RUNS):      
    total_time+=(run_times[run])
    print('Run %s Total: %s' % (run, round(run_times[run], 2)))
    
print('Total Training Runtime: %s' % round(total_time, 2))

print('\nRunning against test data:')
# Run on the test data
test_accuracy = []
test_err = []
_confusions = []

steps = test_size // 100
for step in range(steps):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * 100) % (test_size - 100)
    batch_data = test_data[offset:(offset + 100), :, :, :]
    batch_labels = test_labels[offset:(offset + 100)]
    feed_dict = {test_data_node: batch_data, test_labels_node: batch_labels}

    try:
        a = s.run(test_prediction, feed_dict=feed_dict)
        test_error, confusions = error_rate(a, batch_labels)
        test_err.append(test_error)
        _confusions.append(confusions)
        _predictions = get_predictions(a, batch_labels)
        accur = s.run(accuracy(_predictions), feed_dict=feed_dict)
        test_accuracy.append(accur)
    except ValueError:
        pass

print("Test Accuracy: %.2f%%" % (sum(test_accuracy)/len(test_accuracy)))
print('Test Error: %.1f%%' % (sum(test_err)/len(test_err)))

# calcualte more precise accuracy
unique, counts = np.unique(np.argmax(test_labels, 1), return_counts=True)

y = [sum(x) for x in zip(*_confusions)]

actual = []
counter=0
position=0
for ix in y:
    for iz in ix:
        if counter == position:
            actual.append(iz)
            counter+=11
        position+=1

# Individual accuracy, by number
print('Digit\t Accuracy')
for item in unique:
    print('%s\t %.2f%%' % (item, actual[item]/counts[item]))
    
print('\nGuessed %s out of %s' % (int(np.sum(actual)), np.sum(counts)))
print('Overall Accuracy: %.2f%%' %(np.sum(actual)/np.sum(counts)))

confusion = []
for ix in y:
    for iy in ix:
        confusion.append(iy)

confusion = np.reshape(confusion, (10, 10)) 

# print some graphics of the model's accuracy
plt.figure(1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion Matrix (Runs: %s, Batch Size: %s)' % (NUM_RUNS, BATCH_SIZE))
plt.grid(False)
plt.xticks(np.arange(NUM_LABELS))
plt.yticks(np.arange(NUM_LABELS))
plt.imshow(confusion, cmap=plt.cm.jet, interpolation='nearest');

for i, cas in enumerate(confusion):
    for j, count in enumerate(cas):
        if count > 0:
            xoff = .07 * len(str(count))
            plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')

# save the confusion matrix to disk
plt.savefig('tensor-confusion-60-1.pdf', bbox_inches='tight', dpi=None,
            facecolor='w', edgecolor='b', orientation='portrait',
            papertype=None, pad_inches=0.25, frameon=None)
