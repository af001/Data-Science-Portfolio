# -*- coding: utf-8 -*-
import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from os.path import basename

# download stopwords from nltk corpus
print('\n[+] Downloading nltk stopword corpus...')
nltk.download('stopwords')

# Load the data
print('\n[+] Loading the data into dataframes...')
train = pd.read_csv(r"reviews/train/traindata.tsv", delimiter="\t")
test = pd.read_csv(r"reviews/test/testdata.tsv", delimiter="\t")
# the 8 tom reviews make it difficult to computer with our current batch size.
# in the future, add a mechanism to predict low volumes of data
unsup = pd.read_csv(r"reviews/train/unsupdata.tsv", delimiter="\t")

# # Inspect the Data
print('\n[+] Inspect the data:')
print('[*] Train:')
print(train.head())
print(train.shape)
print('\n[*] Test:')
print(test.head())
print(test.shape)
print('\n[*] Unsup:')
print(unsup.head())
print(unsup.shape)

# The reviews are rather long, so we won't be using all of the text to train our model. Using all of the text would increase our training to a longer timeframe than I would rather give to this project, but it should make the predictions more accurate.
# Inspect the reviews
print('\n[+] Inspect a few reviews:')
for i in range(3):
    print(train.review[i])
    print()

# Check for any null values
print('[+] Check for null values:')
print('[*] Train:\n', train.isnull().sum())
print('[*] Test:\n', test.isnull().sum())
print('[*] Unsup:\n', unsup.isnull().sum())

# # Clean and Format the Data
def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''
    
    extra_words = ['dick', 'ginger', 'hollywood', 'jack', 'jill', 'john', 'karloff', 'kudrow', 'orson', 
                  'peter', 'tcp', 'tom', 'toni', 'welles', 'william', 'wolheim', 'nikita', 'cant', 'didnt'
                  'doesnt', 'dont', 'goes', 'isnt', 'hes', 'shes', 'thats', 'theres', 'theyre', 'wont', 
                  'youll', 'youre', 'youve', 'br', 've', 're', 'vs']
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english") + extra_words)
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"   ", " ", text) # Remove any extra spaces
    text = re.sub(r"  ", " ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Return a list of words
    return(text)

# Clean the training and testing reviews (plus tom and unsup reviews)
train_clean = []
for review in train.review:
    train_clean.append(clean_text(review))

test_clean = []
for review in test.review:
    test_clean.append(clean_text(review))
    
unsup_clean = []
for review in unsup.review:
    unsup_clean.append(clean_text(review))

# Inspect the cleaned reviews
print('\n[+] Cleaned reviews:')
for i in range(3):
    print(train_clean[i], '\n')

# Tokenize the reviews
all_reviews = train_clean + test_clean + unsup_clean
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("[+] Fit is complete.")

train_seq = tokenizer.texts_to_sequences(train_clean)
print("train_seq is complete.")

test_seq = tokenizer.texts_to_sequences(test_clean)
print("test_seq is complete")

unsup_seq = tokenizer.texts_to_sequences(unsup_clean)
print("unsup_clean is complete")

# Find the number of unique tokens
word_index = tokenizer.word_index
print("\n[+] Words in index: %d" % len(word_index))

# Inspect the reviews after they have been tokenized
for i in range(3):
    print('\n[+] Tokenized Reviews:\n', train_seq[i])

# Find the length of reviews
lengths = []
for review in train_seq:
    lengths.append(len(review))

for review in test_seq:
    lengths.append(len(review))
    
for review in unsup_seq:
    lengths.append(len(review))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])

lengths.counts.describe()

print('\n[+] Reveiw counts, by percentile:')
print('80th Percentile: %s' % np.percentile(lengths.counts, 80))
print('85th Percentile: %s' %np.percentile(lengths.counts, 85))
print('90th Percentile: %s' %np.percentile(lengths.counts, 90))
print('95th Percentile: %s' %np.percentile(lengths.counts, 95))

# Pad and truncate the questions so that they all have the same length.
# This covers the majority of reviews, but results in slower training
# We want to cover at least 95% of the review lengths. Based on the
# above calculations.
max_review_length = 290

print('\n[+] Pad the reviews:')
train_pad = pad_sequences(train_seq, maxlen = max_review_length)
print("train_pad is complete")

test_pad = pad_sequences(test_seq, maxlen = max_review_length)
print("test_pad is complete")

unsup_pad = pad_sequences(unsup_seq, maxlen = max_review_length)
print("unsup_pad is complete")

# Inspect the reviews after padding has been completed. 
print('\n[+] Verify Padding:')
for i in range(3):
    print(train_pad[i,:100], '\n')

# Creating the training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(train_pad, train.sentiment, test_size = 0.10, random_state = 2)
x_test = test_pad
y_test = test.sentiment
x_unsup = unsup_pad

# Inspect the shape of the data
print('[+] Validate data shapes:')
print('Train: ', x_train.shape)
print('Valid: ', x_valid.shape)
print('Test: ', x_test.shape)
print('Unsup: ', x_unsup.shape)

# # Build and Train the Model
def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

# get test batches
def get_test_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def get_unsup_batches(x, batch_size):
    '''Create the batches for the testing data'''
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]

# define LSTM cells and dropout
def lstm_cell():
    cell = tf.contrib.rnn.LSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)

# build the rnn, set biases, layers, etc
def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, 
              dropout, learning_rate, multiple_fc, fc_units):
    '''Build the Recurrent Neural Network'''

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Create the embeddings
    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple = True)
    
    # Set the initial state
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 initial_state=initial_state)        
    # Create the fully connected layers
    with tf.name_scope("fully_connected"):
        
        # Initialize the weights and biases
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        
        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                                  num_outputs = fc_units,
                                                  activation_fn = tf.sigmoid,
                                                  weights_initializer = weights,
                                                  biases_initializer = biases)
        dense = tf.contrib.layers.dropout(dense, keep_prob)
        
        # Depending on the iteration, use a second fully connected layer
        if multiple_fc == True:
            dense = tf.contrib.layers.fully_connected(dense,
                                                      num_outputs = fc_units,
                                                      activation_fn = tf.sigmoid,
                                                      weights_initializer = weights,
                                                      biases_initializer = biases)
            dense = tf.contrib.layers.dropout(dense, keep_prob)
    
    # Make the predictions
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense, 
                                                        num_outputs = 1, 
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer = weights,
                                                        biases_initializer = biases)
        tf.summary.histogram('predictions', predictions)
    
    # Calculate the cost
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    
    # Train the model
    with tf.name_scope('train'):    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Determine the accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'keep_prob', 'initial_state', 'final_state','accuracy',
                    'predictions', 'cost', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph

def train(model, epochs, log_string):
    '''Train the RNN'''
    
    # modify config for GPU so that it doesn't allocate all memory at once
    # not needed if using a CPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # create a tensorflow session and save results
    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        logdir = r"tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)
        
        # Used to determine when to stop the training early
        valid_loss_summary = []
        
        # Keep track of which batch iteration is being trained
        iteration = 0

        print("\n[+] Training Model: {}".format(log_string))

        print('[+] Writing training logs to ./log/train/{}'.format(log_string))
        print('[+] Writing validation logs to ./log/valid/{}'.format(log_string))
        train_writer = tf.summary.FileWriter('./logs/log/train/{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('./logs/log/valid/{}'.format(log_string))

        for e in range(epochs):
            state = sess.run(model.initial_state)
            
            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []

            with tqdm(total=len(x_train)) as pbar:
                for _, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):
                    feed = {model.inputs: x,
                            model.labels: y[:, None],
                            model.keep_prob: dropout,
                            model.initial_state: state}
                    summary, loss, acc, state, _ = sess.run([model.merged, 
                                                             model.cost, 
                                                             model.accuracy, 
                                                             model.final_state, 
                                                             model.optimizer], 
                                                            feed_dict=feed)                
                    
                    #Write summary to Tensorboard
                    writer.add_summary(summary, e)
            
                    # Record the loss and accuracy of each training batch
                    train_loss.append(loss)
                    train_acc.append(acc)
                    
                    # Record the progress of training
                    train_writer.add_summary(summary, iteration)
                    
                    iteration += 1
                    pbar.update(batch_size)
            
            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc) 

            val_state = sess.run(model.initial_state)
            with tqdm(total=len(x_valid)) as pbar:
                for x, y in get_batches(x_valid, y_valid, batch_size):
                    feed = {model.inputs: x,
                            model.labels: y[:, None],
                            model.keep_prob: 1,
                            model.initial_state: state}
                    summary, batch_loss, batch_acc, val_state = sess.run([model.merged, 
                                                                          model.cost, 
                                                                          model.accuracy, 
                                                                          model.final_state], 
                                                                         feed_dict=feed)
                    
                    # Record the validation loss and accuracy of each epoch
                    val_loss.append(batch_loss)
                    val_acc.append(batch_acc)
                    pbar.update(batch_size)
            
            # Average the validation loss and accuracy of each epoch
            avg_valid_loss = np.mean(val_loss)    
            avg_valid_acc = np.mean(val_acc)
            valid_loss_summary.append(avg_valid_loss)
            
            # Record the validation data's progress
            valid_writer.add_summary(summary, iteration)

            # Print the progress of each epoch
            print("Epoch: {}/{}".format(e, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc),
                  "Valid Loss: {:.3f}".format(avg_valid_loss),
                  "Valid Acc: {:.3f}".format(avg_valid_acc))

            # Stop training if the validation loss does not decrease after 3 epochs
            if avg_valid_loss > min(valid_loss_summary):
                print("No Improvement.")
                stop_early += 1
                if stop_early == 3:
                    break   
            
            # Reset stop_early if the validation loss finds a new low
            # Save a checkpoint of the model
            else:
                print("New Record!")
                stop_early = 0
                checkpoint = "c:/Users/Anton/Desktop/checkpoint/sentiment_{}.ckpt".format(log_string)
                saver.save(sess, checkpoint)

# The default parameters of the model
n_words = len(word_index)
embed_size = 300
batch_size = 100
lstm_size = 128
num_layers = 2
dropout = 0.5
learning_rate = 0.001
epochs = 100
multiple_fc = False
fc_units = 256

# Train the model with the desired tuning parameters
for lstm_size in [64,128]:
    for multiple_fc in [True, False]:
        for fc_units in [128, 256]:
            log_string = 'ru={},fcl={},fcu={}'.format(lstm_size,
                                                      multiple_fc,
                                                      fc_units)
            model = build_rnn(n_words = n_words, 
                              embed_size = embed_size,
                              batch_size = batch_size,
                              lstm_size = lstm_size,
                              num_layers = num_layers,
                              dropout = dropout,
                              learning_rate = learning_rate,
                              multiple_fc = multiple_fc,
                              fc_units = fc_units)            
            train(model, epochs, log_string)

def make_predictions(lstm_size, multiple_fc, fc_units, checkpoint):
    '''Predict the sentiment of the testing data'''
    
    # allow modification of global variables
    global batch_accuracy
    global count
    
    # Record all of the predictions and mean accuracy
    all_preds = []
    unsup_preds = []
    accur = []

    model = build_rnn(n_words = n_words, 
                      embed_size = embed_size,
                      batch_size = batch_size,
                      lstm_size = lstm_size,
                      num_layers = num_layers,
                      dropout = dropout,
                      learning_rate = learning_rate,
                      multiple_fc = multiple_fc,
                      fc_units = fc_units) 
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        # Load the model
        saver.restore(sess, checkpoint)
        test_state = sess.run(model.initial_state)
        for _, (x,y) in enumerate(get_test_batches(x_test, y_test, batch_size), 1):
            feed = {model.inputs: x,
                    model.labels: y[:, None],
                    model.keep_prob: 1,
                    model.initial_state: test_state}
            predictions, acc = sess.run([model.predictions, model.accuracy], feed_dict=feed)
        
            # store the accuracy
            accur.append(acc)
        
            for pred in predictions:
                all_preds.append(float(pred))
        
        b = basename(checkpoint)
        filename = os.path.splitext(b)[0]
        
        # store the overall accuracy
        print('Predictive Accuracy for %s: %s' % (filename, sum(accur)/float(len(accur))))
        batch_accuracy[count] = sum(accur)/float(len(accur))
        count+=1
        
        # also predict the unsup / tom review data  
        for _, x in enumerate(get_unsup_batches(x_unsup, batch_size), 1):
            feed = {model.inputs: x,
                    model.keep_prob: 1,
                    model.initial_state: test_state}
            predictions = sess.run(model.predictions, feed_dict=feed)
            
            for pred in predictions:
                unsup_preds.append(float(pred))     
        
    sess.close()
    return all_preds, unsup_preds

print('\n[+] Run test and unsupervised data:')
# used to hold values of accuracy, by parameters, to determine best model
# count will == checkpoint. This variable is not implemented below, but
# should create some sort of csv that holds values for each method
batch_accuracy = {}
count = 0

# show 4 checkpoints. Ideally this would be 8 to include the ru=64,fcl=True,fcu=128
# models, but there appears to be a bug in tensorflow that creates an invalid argument
# when chaging the shape of the model using ru=64. 
checkpoint1 = r"c:/Users/Anton/Desktop/checkpoint/sentiment_ru=128,fcl=False,fcu=256.ckpt"
checkpoint2 = r"c:/Users/Anton/Desktop/checkpoint/sentiment_ru=128,fcl=False,fcu=128.ckpt"
checkpoint3 = r"c:/Users/Anton/Desktop/checkpoint/sentiment_ru=128,fcl=True,fcu=256.ckpt"
checkpoint4 = r"c:/Users/Anton/Desktop/checkpoint/sentiment_ru=128,fcl=True,fcu=128.ckpt"

# Make predictions using the best 3 models
predictions1, unsup1 = make_predictions(128, False, 256, checkpoint1)
predictions2, unsup2 = make_predictions(128, False, 128, checkpoint2)
predictions3, unsup3 = make_predictions(128, True, 256, checkpoint3)
predictions4, unsup4 = make_predictions(128, True, 128, checkpoint4)

# from the above, we would include the below saved checkpoints to determine the best configuration.
# until the bug is fixed, these are commented out and removed from the remaining code. 
#predictions5, unsup5 = make_predictions(64, False, 128, checkpoint5)
#predictions6, unsup6 = make_predictions(64, False, 256, checkpoint6)
#predictions7, unsup7 = make_predictions(64, True, 128, checkpoint7)
#predictions8, unsup8 = make_predictions(64, True, 256, checkpoint8)

# Average the best three predictions
predictions_combined = (pd.DataFrame(predictions1) + pd.DataFrame(predictions2) + pd.DataFrame(predictions3) + pd.DataFrame(predictions4))/4
unsup_combined = (pd.DataFrame(unsup1) + pd.DataFrame(unsup2) + pd.DataFrame(unsup3) + pd.DataFrame(unsup4))/4

def write_submission(predictions, string, data):
    '''write the predictions to a csv file'''
    if data == 'test':
        submission = pd.DataFrame(data={"id":test["id"], "predict sentiment":predictions, "sentiment":test["sentiment"]})
    elif data == 'unsup':
        submission = pd.DataFrame(data={"id":unsup["id"], "predict sentiment":predictions, "sentiment":np.nan})
        
    submission.to_csv("submission_{}.csv".format(string), index=False, quoting=3)
    print('\n[+] Test Predictions for %s [Top 3]:\n' % string, submission.head(3))

# write the results for each file to disk for training
write_submission(predictions1, "test_ru=128,fcl=False,fcu=256", 'test') 
write_submission(predictions2, "test_ru=128,fcl=False,fcu=128", 'test') 
write_submission(predictions3, "test_ru=128,fcl=True,fcu=256", 'test') 
write_submission(predictions4, "test_ru=128,fcl=True,fcu=128", 'test') 
write_submission(predictions_combined.ix[:,0], "test_combined", 'test') 
                        
# write the results for each file to disk for unsup
write_submission(unsup1, "unsup_ru=128,fcl=False,fcu=256", 'unsup') 
write_submission(unsup2, "unsup_ru=128,fcl=False,fcu=128", 'unsup') 
write_submission(unsup3, "unsup_ru=128,fcl=True,fcu=256", 'unsup') 
write_submission(unsup4, "unsup_ru=128,fcl=True,fcu=128", 'unsup') 
write_submission(unsup_combined.ix[:,0], "unsup_combined", 'unsup') 
              
print('\n[+] Overall accuracy:')

for acc in batch_accuracy:
    if acc == 0:
        print('Method1: %s' % batch_accuracy[acc])
    elif acc == 1:
        print('Method2: %s' % batch_accuracy[acc])
    elif acc == 2:
        print('Method3: %s' % batch_accuracy[acc])
    elif acc == 3:
        print('Method4: %s' % batch_accuracy[acc])
