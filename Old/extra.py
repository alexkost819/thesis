def identify_labels(filenames):
    """
    Create a tensor of labels from the filenames
    """
    labels = []
    for file in filenames:
        psi = int(file[PSI_BEGIN:PSI_END])
        if psi < PSI_NOM_MIN:
            labels.append(LABEL_UNDER)
        elif psi < PSI_NOM_MAX:
            labels.append(LABEL_NOM)
        else:
            labels.append(LABEL_OVER)

    return labels

def get_csv_binary(file):
    """
    Get binary data from CSV files
    """
    csv_data = np.genfromtxt(file, delimiter=',')
    pdb.set_trace()
    features = csv_data[1:n_features+1]
    label = csv_data[-1][0]

    return features.tobytes(), label.tobytes()


def write_to_tfrecord(label, features, tfrecord_file):
    """
    Convert csv binary to TFRecord
    CREDIT: http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape])),
    }))
    writer.write(example.SerializeToString())
    writer.close()

def read_from_tfrecord(filenames):
    """

    CREDIT: http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf

    """
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
    features={
    'label': tf.FixedLenFeature([], tf.string),
    'shape': tf.FixedLenFeature([], tf.string),
    'image': tf.FixedLenFeature([], tf.string),
    }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, shape)
    label = tf.cast(tfrecord_features['label'], tf.string)
    return label, shape, image


class RNN_Model(object):
    """
    RNN_Model is a class that contains the computational graph used in main_rnn.py
    """
    def __init__(self):

        """ Step 0: Define training and model parameters """
        learning_rate = 0.001           # learning rate used in optimizer
        num_epochs = 1000               # number of times we go through all data
        training_iters = num_epochs         # TODO: MORE HERE
        batch_size = 64                 # number of examples in a single batch
        display_step = 100              # Every _ steps, save to tensorboard and display info

        n_input = 28                    # MNIST data input (img shape: 28*28)
        n_steps = 28                    # timesteps
        n_hidden = 128                  # hidden layer num of features
        n_classes = 3                   # classifications: over, under, correct pressure

        """ Create tensors out of the input and output (data, labels) """
        self.input = config['input']
        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        max_grad_norm = config['max_grad_norm']
        self.batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        num_classes = config['num_classes']
        """Place holders"""
        self.input = tf.placeholder(tf.float32, [None, sl], name = 'input')
        self.labels = tf.placeholder(tf.int64, [None], name='labels')
        self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')


    def write_to_tfrecord(label, shape, binary_image, tfrecord_file):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape])),
    'image':tf.train.Feature(bytes_list=tf.train.BytesList(
    value=[binary_image]))
    }))
    writer.write(example.SerializeToString())
    writer.close()


    def write_data_to_tfrecord(record_name):
    """
    Writes data to TFRecord
    CREDIT: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
    """

    train_filename = record_name  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print 'Train data: {}/{}'.format(i, len(train_addrs))
            sys.stdout.flush()
        # Load the image
        img = load_image(train_addrs[i])
        label = train_labels[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

    def idk():

    # # Extract data from CSV
    # labels = []
    # binary = []
    # for file in filenames:
    #     # Identify the label by PSI
    #     psi = int(file[PSI_BEGIN:PSI_END])
    #     if psi < PSI_NOM_MIN:
    #         labels.append(LABEL_UNDER)
    #     elif psi < PSI_NOM_MAX:
    #         labels.append(LABEL_NOM)
    #     else:
    #         labels.append(LABEL_OVER)

    #     # Convert the csv to binary
    #     with open(file, "r") as f:
    #         binary.append(f.read())

      
    #     writer = tf.python_io.TFRecordWriter(file)
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file])),
    #         'y'       : tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tobytes()]))
    #         }))
    #     writer.write(example.SerializeToString())
    #     writer.close()

    def stacked_rnn_model(x, weights, biases, time_steps=n_steps, depth=2):
        """
        Stacked RNN model
        """
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)

        stacked_lstm = rnn_cell.MultiRNNCell([rnn_cell] * depth)
        x_split = tf.split(batch_size, time_steps, x)
        outputs, states = rnn.rnn(stacked_lstm, x_split)
        return tf.matmul(outputs[-1], weights) + biases

    def rnn_model(x, weights, biases):
        """
        RNN model (simple)
        """
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)

        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # last row of outputs is last row of output
        return tf.matmul(outputs[-1], weights) + biases

    def classifier(self):
        """
        Stretch goal: Make the RNN model nice as fuck
        This is the softmax classification step
        What happens here:
        1) Scopes created for nice tensorboard presentation
        2) softmax_w and softmax_b variables defined
        3) logits calculated, then used to identify loss and cost (what's the difference?)
        """

        # We're getting SLICES of each level, like a SANDWICH
        # https://stackoverflow.com/questions/43563609/how-tf-transpose-works-in-tensorflow
        # BEFORE: [batch_size x max_time x n_hidden]
        # AFTER:  [max_time x batch_size x n_hidden]
        # DIMS:   [depth x rows x columns]
        # outputs = tf.transpose(outputs, [1, 0, 2])
        # last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
        # Softmax classification


        with tf.name_scope("Softmax") as scope:
            with tf.variable_scope("Softmax_params"):
                # get_variable vs Variable: https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
                softmax_w = tf.get_variable("softmax_w", [n_hidden, n_classes])
                softmax_b = tf.get_variable("softmax_b", [n_classes])

                # xw_plus b = output * softmax_w + softmax_b
                logits = tf.nn.xw_plus_b(self.output, softmax_w, softmax_b)

                # use sparse Softmax because we have mutually exclusive classes
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.labels,name = 'softmax')
                self.cost = tf.reduce_sum(self.loss) / self.batch_size
        with tf.name_scope("Evaluating_accuracy") as scope:
            correct_prediction = tf.equal(tf.argmax(logits,1),self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            h1 = tf.summary.scalar('accuracy',self.accuracy)
            h2 = tf.summary.scalar('cost', self.cost)



        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
        print('Finished computation graph')
    