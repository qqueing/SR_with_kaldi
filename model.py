#__AUTHOR__ : qqueing


import tensorflow as tf
import numpy as np

from tf_block import BatchNorm as bn_block
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

class Model(object):

    def __init__(self):
        self.graph = tf.Graph()

    def build_model(self,input_vector_length,  filter_sizes, kernel_sizes, num_classes,input_dim,embeded_sizes,learning_rate):

        with self.graph.as_default():

            self.num_classes = num_classes

            # Placehodlers for regular data
            self.input_x = tf.placeholder(tf.float32, [None, input_vector_length,input_dim], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            # placeholder for parameter
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.Phase = tf.placeholder(tf.bool, name="Training_Phase")

            l2_loss = tf.constant(0.0)

            # Mean nomalization using batch nomalization
            with tf.variable_scope("input"):
                h = bn_block(self.input_x, decay=0.9,scale=False, is_training=self.Phase)

            #Frame level information Layer
            prev_dim = input_dim
            for i, (kernel_size,filter_size) in enumerate(zip(kernel_sizes,filter_sizes)):
                with tf.variable_scope("frame_level_infor_layer-%s" % i):
                    if kernel_size == 0:
                        kernel_shape = [conv.shape[1]._value, prev_dim, filter_size]
                    else:
                        kernel_shape = [kernel_size, prev_dim, filter_size]
                    W = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[filter_size]), name="b")

                    conv = tf.nn.conv1d(h, W, stride=1, padding="VALID", name="conv-layer-%s" % i)

                    # Apply BN and nonlinearity
                    conv = bn_block(conv, decay=0.9, scale=True, is_training=self.Phase)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    prev_dim = filter_size

                    #Apply dropout
                    if i != len(kernel_sizes)-1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)


            #Statistic pooling
            tf_mean,tf_var = tf.nn.moments(h,1)
            h = tf.concat([tf_mean,tf_var], 1)
            prev_dim = prev_dim *2


            #Embedding Layer
            for i, out_dim in enumerate(embeded_sizes):

                with tf.variable_scope("embed_layer-%s" % i):
                    W = tf.Variable(tf.truncated_normal([prev_dim,out_dim], stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")
                    h = tf.nn.xw_plus_b(h, W, b, name="scores")

                    #Make for output
                    if i  == 0 :
                        self.embedding_1_non_precssing = h
                    elif i == 1:
                        self.embedding_2_non_precssing = h

                    h = bn_block(h, decay=0.9, scale=True, is_training=self.Phase)
                    h = tf.nn.relu(h, name="relu")

                    prev_dim = out_dim
                    if i != len(embeded_sizes)-1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)


            # Softmax
            with tf.variable_scope("output"):
                W = tf.get_variable("W",shape=[prev_dim, num_classes], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

                #Apply L2 loss
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.nn.xw_plus_b(h, W, b, name="scores")
                predictions = tf.argmax(scores, 1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = self.input_y)

            self.loss = tf.reduce_mean(losses) + 0.15 * l2_loss



            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def run(self, x):


        with tf.Session(graph=self.graph,config=tf.ConfigProto(log_device_placement=True)) as sess:
            train_data, train_labels, test_data, test_labels = self.create_data(x)

            # Inizitalization
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            batch_size = 64
            num_epochs = 50
            drop_out_prob = 0.5

            steps_per_epoche = int(len(train_labels) / batch_size)
            num_steps = steps_per_epoche * num_epochs
            epoch_num = 0
            tic()
            for step in range(num_steps):
                # Shuffle the data in each epoch
                if (step % steps_per_epoche == 0):

                    #
                    toc()
                    tic()
                    shuffle_indices = np.random.permutation(np.arange(len(train_data)))
                    train_data = train_data[shuffle_indices]
                    train_labels = train_labels[shuffle_indices]
                    print("epoche number %d" % epoch_num)
                    sum_accuracy_out = 0.0;
                    test_data_length = 0.0
                    for offset in range(0,test_labels.shape[0],batch_size):
                        batch_data = test_data[offset:(offset + batch_size), :]
                        batch_labels = test_labels[offset:(offset + batch_size), :]
                        feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 1.0, self.Phase: False}
                        accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
                        sum_accuracy_out += accuracy_out*len(batch_data)
                        test_data_length += len(batch_data)
                    print('Test accuracy: %.3f' % (sum_accuracy_out/test_data_length))
                    epoch_num += 1
                    saver.save(sess, 'data/tf_dump/model.ckpt',global_step=epoch_num)

                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]

                feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: drop_out_prob,
                             self.Phase: True}
                _, l, accuracy_out = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                #print('Training Accuracy: %.3f and loss  : %.3f' % (accuracy_out, l))

            # Last test
            sum_accuracy_out = 0.0;
            test_data_length = 0.0
            for offset in range(0, test_labels.shape[0], batch_size):
                batch_data = test_data[offset:(offset + batch_size), :]
                batch_labels = test_labels[offset:(offset + batch_size), :]
                feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 1.0,
                             self.Phase: False}
                accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
                sum_accuracy_out += accuracy_out * len(batch_data)
                test_data_length += len(batch_data)
            print('Test accuracy: %.3f' % (sum_accuracy_out / test_data_length))

    def eval(self, x):

        batch_size = 64

        with tf.Session(graph=self.graph) as sess:
            train_data, train_labels, test_data, test_labels = self.create_data(x)

            # Inizitalization
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('./data/tf_dump/'))

            sum_accuracy_out = 0.0;
            test_data_length = 0.0
            for offset in range(0, test_labels.shape[0], batch_size):
                batch_data = test_data[offset:(offset + batch_size), :]
                batch_labels = test_labels[offset:(offset + batch_size), :]
                feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 1.0,
                             self.Phase: False}
                accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
                sum_accuracy_out += accuracy_out * len(batch_data)
                test_data_length += len(batch_data)
            print('Test accuracy: %.3f' % (sum_accuracy_out / test_data_length))

    def make_embedding(self,x):

        test_data = []
        test_id = []
        test_label = []
        for datum in x:
            test_id.append(datum)
            test_data.append(x[datum])
            test_label.append([0] * self.num_classes)
        batch_size = 64

        test_data = np.asarray(test_data, dtype=np.float32)
        test_label = np.asarray(test_label, dtype=np.float32)
        outputs ={}
        outputs['key'] =test_id
        outputs['embed'] = []

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('./data/tf_dump/'))

            for offset in range(0, test_data.shape[0], batch_size):
                batch_data = test_data[offset:(offset + batch_size), :]
                batch_labels = test_label[offset:(offset + batch_size), :]
                feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 1.0,
                             self.Phase: False}
                embedding_2 = sess.run(self.embedding_2_non_precssing, feed_dict=feed_dict)
                outputs['embed'].append(embedding_2)

        outputs['embed'] = np.concatenate(outputs['embed'])

        return outputs




    def make_data(self, x,input_seq_length =200):
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        for datum in x:
            idx_list = datum['idx']
            label = [0] * self.num_classes
            label[datum["y"]] = 1
            for idx, start_idx in enumerate(idx_list):
                if idx < 1 and len(idx_list)>1 :
                    if len(datum['speech']) > input_seq_length:
                        test_data.append(datum['speech'][start_idx:start_idx+input_seq_length])
                        test_labels.append(label)
                    else:
                        temp_data = np.zeros((input_seq_length, 20))
                        temp_data[0:len(datum['speech'])][:] = datum['speech'][:][:]
                        test_data.append(temp_data)
                        test_labels.append(label)
                        #print('sktp')


                else:
                    if len(datum['speech'])> input_seq_length:
                        train_data.append(datum['speech'][start_idx:start_idx + input_seq_length])
                        train_labels.append(label)
                    else:
                        temp_data = np.zeros((input_seq_length, 20))
                        temp_data[0:len(datum['speech'])][:] = datum['speech'][:][:]
                        train_data.append(temp_data)
                        train_labels.append(label)
                        #print('sktp')



        train_data = np.array(train_data, dtype=np.float32)
        test_data = np.array(test_data, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.float32)
        test_labels = np.asarray(test_labels, dtype=np.float32)

        return [train_data, train_labels, test_data, test_labels]

    def create_data(self, x):
        train_data, train_labels, test_data, test_labels = self.make_data(x)

        shuffle_indices = np.random.permutation(np.arange(len(train_data)))
        train_data = train_data[shuffle_indices]
        train_labels = train_labels[shuffle_indices]

        return train_data, train_labels, test_data, test_labels


