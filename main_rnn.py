"""
Created on 24 June 2017

@author: Alex Kost

@description: Main python code file for Applying RNN as a TPMS

"""

import logging
import os
import tensorflow as tf
import time

""" Constants """
DEFAULT_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
SIM_DATA_PATH = 'Data/simulation'

# Model Parameters
num_epochs = 1                  # number of times we go through all data
learning_rate = 1e-3            # learning rate used in optimizer
n_hidden = 4                    # number of features per hidden layer in LSTM
n_classes = 3                   # classifications: under, nominal, over pressure
n_features = 3                  # sprung_accel, unsprung_accel, sprung_height

#n_layers = 2                   # number of hidden layers in LSTM model
#dropout_prob = 0.5              # dropout probability

batch_size = 5                  # number of examples in a single batch
display_step = 10               # Every _ steps, save to tensorboard and display info

# Magic Constant Numbers
SEQUENCE_LENGTH = int(1.75 / .001) + 1      # sec/sec
CSV_N_COLUMNS = 5
LABEL_UNDER = 0
LABEL_NOM = 1
LABEL_OVER = 2


def split_data(data, val_size=0.2, test_size=0.2):
    """
    Spit all the data we have into training, validating, and test sets.
    By default, 64/16/20 split (20% of 80% = 16%)
    Credit: https://www.slideshare.net/TaegyunJeon1/electricity-price-forecasting-with-recurrent-neural-networks
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train = data.iloc[:nval]
    df_val = data.iloc[nval:ntest]
    df_test = data.iloc[ntest:]

    return df_train, df_val, df_test

def create_filename_list(data_dir):
    """
    Create filename queue out of files in data_dir
    """
    # Identify CSV files in directory
    filenames = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                rel_filepath = os.path.join(root, file)
                abs_filepath = os.path.abspath(rel_filepath)
                filenames.append(abs_filepath)

    return filenames

def create_filename_queue(filenames):
    """
    Create filename queue out of csv files
    """
    filename_queue = tf.train.string_input_producer(
        string_tensor=filenames,
        num_epochs=num_epochs,
        shuffle=False)

    return filename_queue

def read_batch_from_queue(filename_queue):
    """ 
    Read CSV data and pack into tensors 
    """
    # Step 1: Read CSV data
    reader = tf.TextLineReader(skip_header_lines=0,
                               name='TextLineReader')

    features, labels = [], []

    # identify record_defaults used for decode_csv
    # default values and types if value is missing
    record_defaults = [[0.0] for _ in range(CSV_N_COLUMNS)]
    record_defaults[-1] = [0]

    for i in range(batch_size):
        _, csv_row = reader.read_up_to(filename_queue, SEQUENCE_LENGTH)
        # content = [time, sprung_accel, unsprung_accel, sprung_height, label]
        content = tf.decode_csv(records=csv_row,
                                record_defaults=record_defaults,
                                name='decode_csv')

        # Parse content
        # content = [time, sprung_accel, unsprung_accel, sprung_height, label]
        ex_features = tf.stack(content[1:n_features+1])
        ex_labels = tf.one_hot(content[-1][0], n_classes)

        # Append each tensor to the list
        features.append(ex_features)
        labels.append(ex_labels)

    # Step 2: Stack lists of N-rank tensors to N+1 rank tensors
    features = tf.stack(features)
    labels = tf.stack(labels)

    return features, labels

    # BEFORE TRANSPOSE, Columns and Rows are reversed
    # BEFORE: [batch_size x input size] (vertical x horizontal dims)
    # we want it this: [batch_size, SEQUENCE_LENGTH, n_features])

def main():
    """
    Sup Main!
    """

    # INPUT PIPELINE
    filenames = create_filename_list(SIM_DATA_PATH)
    filename_queue = create_filename_queue(filenames)

    # FEATURES AND LABELS
    # For dynamic_rnn, must have input be in certain shape
    # BEFORE: [batch_size x input_size x max_time]
    # AFTER:  [batch_size x max_time x input_size]
    # DIMS:   [depth x rows x columns]
    x, y = read_batch_from_queue(filename_queue)
    x = tf.transpose(x, [0, 2, 1])

    # MODEL
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
    # dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout_prob)
    # multi_cell = tf.contrib.rnn.MultiRNNCell([dropout] * n_layers)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    # We transpose the output to switch batch size with sequence size.
    # http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
    outputs = tf.transpose(outputs, [1, 0, 2])


    # put the outputs into a classifier
    # xw_plus b = output * softmax_w + softmax_b
    weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
    biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

    pred = tf.layers.dense(outputs[-1], n_classes, name='logits')
    #pred = tf.nn.xw_plus_b(outputs[-1], weights, biases)

    # TRAINING PARAMETERS
    # Cross-Entropy: "measuring how inefficient our predictions are for describing the truth"
    # http://colah.github.io/posts/2015-09-Visual-Information/
    # https://stackoverflow.com/questions/41689451/valueerror-no-gradients-provided-for-any-variable
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
    # Reduce Mean: Computes the mean of elements across dimensions of a tensor.  
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # EVALUATE OUR MODEL
    # tf.argmax = returns index of the highest entry in a tensor along some axis.
    # So here, tf.equal is comparing predicted label to actual label, returns list of bools
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # tf.cast coverts bools to 1 and 0, tf.reduce_mean finds the mean of all values in the list
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    """ Step 2: Set up Tensorboard Saver """
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())

    """ Step 3: Train the RNN """
    with tf.Session() as sess:
        # Initialization
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 1

        logger.info("The training shall begin.")
        try:
            while not coord.should_stop():
                # Identify batches
                batch_x, batch_y = sess.run([x, y])

                # Train with batches defined above
                sess.run(optimizer, feed_dict={
                    x: batch_x,
                    y: batch_y
                })

                """Step 3.2: Display training status and save model to TB """
                if step % display_step == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
                    loss = cost.eval(feed_dict={x: batch_x, y: batch_y})
                    logger.info('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(step * batch_size, loss, acc))
                
                step += 1
                logger.debug('Step %d', step)

        except tf.errors.OutOfRangeError:
            logger.info('Done training, epoch reached')

        # Conclude training
        logger.info("The training is done.")
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(DEFAULT_FORMAT)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    main()
