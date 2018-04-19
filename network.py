
# Original base for network.py from Kwang Moo Yi for CSC486B/586B,
# released under MIT license
# Modified by Austin Hendy, Daria Sova, Maxwell Borden, and Jordan Patterson

import os, h5py, IPython
import numpy as np
import tensorflow as tf
from tqdm import trange

from config import get_config, print_usage
from utils.preprocessing import package_data
from layerutils import fcl, convl

class Network:
    def __init__(self, x_shp, lstm_x_shp, config, speed_x_shp):

        self.config = config

        # Get shape
        self.x_shp = x_shp
        self.lstm_x_shp = lstm_x_shp
        self.speed_x_shp = speed_x_shp
        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_writer(self):
        """Build writers and savers for the model"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")

    def _build_summary(self):
        """Build summary operations"""

        # Merge all the summary op
        self.summary_op = tf.summary.merge_all()

    def _build_placeholder(self):
        """Build placeholders."""

        # Get shape for placeholder
        x_in_shp = (None, *self.x_shp[1:])
        lstm_x_in_shp = (None, *self.lstm_x_shp[1:])
        speed_x_shp = (None, *self.speed_x_shp[1:])
        print("Shapes ", lstm_x_in_shp, speed_x_shp)
        # Create Placeholders for inputs
        self.seg_x = tf.placeholder(tf.float32, shape=x_in_shp, name="seg_x_in")
        self.seg_y = tf.placeholder(tf.int64, shape=x_in_shp[:-1], name="seg_y_in")
        
        self.lstm_x = tf.placeholder(tf.float32, shape=lstm_x_in_shp, name="lstm_x_in")
        self.lstm_y = tf.placeholder(tf.int64, shape=x_in_shp[:-1], name="lstm_y_in")
        
        self.lstm_speed_x = tf.placeholder(tf.float32, shape=speed_x_shp, name="lstm_speed_x")
        self.lstm_speed_y = tf.placeholder(tf.int64, shape=speed_x_shp[0], name="lstm_speed_y")
 
    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):
            # Create placeholders for saving mean, range to a TF variable for
            # easy save/load. Create these variables as well.
            self.n_mean_in = tf.placeholder(tf.float32, shape=())
            self.n_range_in = tf.placeholder(tf.float32, shape=())
            # Make the normalization as a TensorFlow variable. This is to make
            # sure we save it in the graph
            self.n_mean = tf.get_variable(
                "n_mean", shape=(), trainable=False)
            self.n_range = tf.get_variable(
                "n_range", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.n_assign_op = tf.group(
                tf.assign(self.n_mean, self.n_mean_in),
                tf.assign(self.n_range, self.n_range_in),
            )

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(
                "global_step", shape=(),
                initializer=tf.zeros_initializer(),
                dtype=tf.int64,
                trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):

            # Compute the accuracy of the Segmentation.
            self.seg_pred = tf.argmax(self.seg_logits, axis=3) # Argmax per pixel
            self.seg_acc = tf.reduce_mean( 
                tf.to_float(tf.equal(self.seg_pred, self.seg_y))
            )

            # TODO: accuracy for LSTM
            self.lstm_pred = tf.argmax(self.lstm_out, axis=1)
            self.lstm_acc = tf.reduce_mean( 
                tf.to_float(tf.equal(self.lstm_pred, self.lstm_speed_y))
            )
            
            # Record summary for accuracies
            tf.summary.scalar("seg_accuracy", self.seg_acc)
            tf.summary.scalar("lstm_accuracy", self.lstm_acc)

            # save best validation accuracy
            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable(
                "best_va_acc", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.acc_assign_op = tf.assign(self.best_va_acc, self.best_va_acc_in)

    def _load_initial_weights(self, sess):
        '''
        Load weights from a file into network.
        Weights taken from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        It is a dict of lists 
        '''

        print("Loading pretrained weights for Alexnet...")
        # load weights from the file
        weights_dict = np.load(self.config.weights_dir, encoding='bytes').item()
        # Loop over all layer names stored in the weights dict
        # dict_keys(['fc6', 'fc7', 'fc8', 'conv3', 'conv2', 'conv1', 'conv5', 'conv4'])
        for op_name in weights_dict:

            with tf.variable_scope("Network", reuse=True):

                with tf.variable_scope(op_name, reuse=True):
                    try:
                        weights, biases = weights_dict[op_name]

                        # Load Weights
                        if op_name == 'fc6':
                            weights = tf.reshape(weights, (6, 6, 256, 4096))
                        elif op_name == 'fc7':
                            weights = tf.reshape(weights, (1, 1, 4096, 4096))
                        var = tf.get_variable('weights', trainable=False)
                        sess.run(var.assign(weights))

                        # Load Biases
                        var = tf.get_variable('biases', trainable=False)
                        sess.run(var.assign(biases))
                        print("Loading: ", op_name)
                    except:
                        pass
                            
        print("Weights loaded.")


    def alexNet(self, x_in):
        '''
        AlexNext implementation based on: https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/alexnet.py
        '''

        print("Building Alexnet into the network...")
        print("Shape of data going into AlexNet: ", x_in.shape)
        
        # Normalize using the above training-time statistics
        cur_in = (x_in - self.n_mean) / self.n_range
        print("Starting shape...", cur_in.shape)

        # 1st Layer Conv1
        cur_in = convl(cur_in, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        cur_in = tf.contrib.layers.max_pool2d(cur_in, [3, 3], 2, padding='VALID')

        # 2nd Layer Conv2
        cur_in = convl(cur_in, 5, 5, 256, 1, 1, groups=2, name='conv2')
        cur_in = tf.contrib.layers.max_pool2d(cur_in, [3, 3], 2, padding='VALID')

        # 3rd Layer Conv3
        cur_in = convl(cur_in, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer Conv4
        cur_in = convl(cur_in, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer Conv5
        cur_in = convl(cur_in, 3, 3, 256, 1, 1, groups=2, name='conv5')
        
        cur_in = tf.contrib.layers.dropout(cur_in,
                                    0.3, is_training=True)

        # Fully connected layers with conv
        cur_in = convl(cur_in, 6, 6, 4096, 1, 1,  padding='VALID', name='fc6')

        cur_in = tf.contrib.layers.dropout(cur_in, 0.3, is_training=True)

        # 8th Layer: FC (w ReLu) -> Dropout (as conv layer)
        cur_in = convl(cur_in, 1, 1, 4096, 1, 1,  padding='VALID', name='fc7')

        cur_in = tf.contrib.layers.dropout(cur_in, 0.3, is_training=True)
        print("Starting shape...", cur_in.shape)

        # 8th Layer: FC and return unscaled activations
        # cur_in = fcl(cur_in, 4096, self.config.num_class, name='fc8')
        print("AlexNet Done.")
        return cur_in

    def _build_model(self):
        """Build Network."""

        # Build the network (use tf.layers)
        with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
            # Normalize using the above training-time statistics
            yshps = [x.value for x in self.seg_y.get_shape()]
            lstm_x_shps = [x.value for x in self.lstm_x.get_shape()]

            alex_in = tf.reshape(self.lstm_x, (
                -1, self.lstm_x_shp[2], self.lstm_x_shp[3], self.lstm_x_shp[4])
            )
            print("Alex_in shape..", alex_in.shape)
 
            # reshape Segmentation output to N classes final dim
            cur_in_seg = self.alexNet(self.seg_x)
            classwise_seg_preds = tf.contrib.layers.conv2d(cur_in_seg, self.config.num_class, [1, 1],
                                   activation_fn=None,
                                   padding="VALID",
                                   biases_initializer=None)

            # Upscale logits to NWHC using nearest neighbor interpolation
            # turns (?,?,?, num_class) back to class scores for each pixel
            # approximates deconvolution layer
            self.seg_logits = tf.image.resize_images(
                images=classwise_seg_preds,
                size=[yshps[1], yshps[2]], 
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            
            print("Shape After Seg Reshape..", self.seg_logits.shape)

            #### #### LSTM #### ####
            lstm_x_shps = [x.value for x in self.lstm_x.get_shape()]
            cur_in_lstm = self.alexNet(alex_in)
            # cur_in_lstm = tf.contrib.layers.conv2d(cur_in_lstm, 2, [1, 1],
            #                        activation_fn=None,
            #                        padding="VALID",
            #                        biases_initializer=None)
            
            # cur_in_lstm = tf.contrib.layers.max_pool2d(cur_in_lstm, [4, 4], 3, padding='VALID')
            cur_in_lstm = tf.reshape(cur_in_lstm, [tf.shape(self.lstm_speed_y)[0], lstm_x_shps[1], -1])

            # put speed data together
            # lstm input should be (N, T, D) shape
            lstm_in_alex_and_movements = tf.concat([cur_in_lstm, self.lstm_speed_x], axis=-1)

            lstm_in_alex_and_movements = tf.reshape(lstm_in_alex_and_movements, 
                (-1, lstm_x_shps[1], 2))

            lstm_out2 = tf.keras.layers.LSTM(64)(lstm_in_alex_and_movements)
            
            # hidden_out = tf.keras.layers.Reshape(lstm_out3,
                                                #  (self.lstm_x_shp[0] * self.lstm_x_shp[1], -1))
            self.lstm_out = tf.layers.Dense(4)(lstm_out2)
            self.kernels_list = [
                _v for _v in tf.trainable_variables() if "kernel" in _v.name]         
                

    def LSTM(self, X):
        print("Running LSTM...")
        # straight, stop, left, right
        num_classes = 4
        # Define weights
        weights = tf.Variable(tf.random_normal([self.config.num_hidden, num_classes]))
        biases = tf.Variable(tf.random_normal([num_classes]))

        # Lstm cell with tensorflow
        lstm = []
        lstm_cell = rnn.BasicRNNCell(self.config.num_hidden)
        lstm += [lstm_cell]
        # Get outputs
        out, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        # Linear activation
        activ = tf.matmul(out[-1], weights) + biases

        return activ


    def train(self, seg_data, lstm_data, speed_data):
        """Training function.

        Parameters
        ----------
        seg_data : tuple of ndarray
            Training data.
            Training labels.
            Validation data.
            Validation labels.

        lstm_data : tuple of ndarray
            Training data.
            Training labels.
            Validation data.
            Validation labels.

        speed_data : tuple ndarray
            Training data.
            Training labels.
            Validation data.
            Validation labels.
        """

        # Unpack
        seg_x, seg_y, seg_x_va, seg_y_va = seg_data
        lstm_x, lstm_y, lstm_x_va, lstm_y_va = lstm_data
        speed_x, speed_y, speed_x_va, speed_y_va = speed_data

        # ----------------------------------------
        # Preprocess data
        x_tr_mean = seg_x.mean()
        x_tr_range = 128.0

        # Report data statistic
        print("Training data before: mean {}, std {}, min {}, max {}".format(
            x_tr_mean, seg_x.std(), seg_x.min(), seg_x.max()
        ))


        
        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            
            self._load_initial_weights(sess)
            tf.keras.backend.set_session(sess)
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            # Assign normalization variables from statistics of the train data
            sess.run(self.n_assign_op, feed_dict={
                self.n_mean_in: x_tr_mean,
                self.n_range_in: x_tr_range,
            })

            # Check if previous train exists
            b_resume = tf.train.latest_checkpoint(self.config.log_dir)

            if b_resume:
                # Restore network from log_dir for curr model
                print("Restoring from {}...".format(
                    self.config.log_dir))

                self.saver_cur.restore(
                    sess,
                    b_resume
                )

                # Restore number of steps so far
                step = self.global_step.eval()
                # Restore best acc
                best_acc = self.best_va_acc.eval()

            else:
                print("Starting from scratch...")
                step = 0
                best_acc = 0

            print("Training...")
            batch_size = config.batch_size
            max_iter = config.max_iter
            # For each epoch
            for step in trange(step, max_iter):

                # Get a random training batch
                ind_cur = np.random.choice(
                    len(seg_x), batch_size, replace=True)
                seg_x_b = np.array([seg_x[_i] for _i in ind_cur])
                seg_y_b = np.array([seg_y[_i] for _i in ind_cur])
                lstm_x_b = np.array([lstm_x[_i] for _i in ind_cur])
                lstm_y_b = np.array([lstm_y[_i] for _i in ind_cur])
                speed_x_b =  np.array([speed_x[_i] for _i in ind_cur])
                speed_y_b =  np.array([speed_y[_i] for _i in ind_cur])

                # Write summary every N iterations as well as the first iteration
                K = self.config.report_freq
                b_write_summary = step % K == 0 and step!=0 or step == 1
                if b_write_summary:
                    fetches = {
                        "optim": self.optim,
                        "summary": self.summary_op,
                        "global_step": self.global_step,
                    }
                else:
                    fetches = {
                        "optim": self.optim,
                    }

                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.seg_x: seg_x_b,
                        self.seg_y: seg_y_b,
                        self.lstm_x: lstm_x_b,
                        self.lstm_y: lstm_y_b,
                        self.lstm_speed_x: speed_x_b,
                        self.lstm_speed_y: speed_y_b,
                    },
                )

               # Write Training Summary if we fetched it (no meta graph)
                if "summary" in res: 
                   self.summary_tr.add_summary(
                       res["summary"], global_step=res["global_step"],
                   )
                   self.summary_tr.flush()

                   # Also save current model to resume when we write the summary.
                   self.saver_cur.save(
                       sess, self.save_file_cur,
                       global_step=self.global_step,
                       write_meta_graph=False,
                   )

                # Validate every N iterations and at the first iteration.
                V = self.config.val_freq
                b_validate = step % V == 0 and step != 0 or step == 1
                if b_validate:
                    res = sess.run(
                        fetches={
                           "acc": self.seg_acc,
                           "summary": self.summary_op,
                           "global_step": self.global_step,
                        },
                        feed_dict={
                            self.seg_x: seg_x_va,
                            self.seg_y: seg_y_va,
                            self.lstm_x: lstm_x_va,
                            self.lstm_y: lstm_y_va,
                            self.lstm_speed_x: speed_x_va,
                            self.lstm_speed_y: speed_y_va,
                        })
                    # Write Validation Summary
                    self.summary_va.add_summary(
                       res["summary"], global_step=res["global_step"],
                    )
                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and best accuracy
                    if res["acc"] > best_acc:
                       best_acc = res["acc"]
                       # Write best acc to TF variable
                       sess.run(self.acc_assign_op, feed_dict={
                           self.best_va_acc_in: best_acc
                       })

                       # Save the best model
                       self.saver_best.save(
                           sess, self.save_file_best,
                           write_meta_graph=False,
                       )

    def test(self, x_te, y_te):
        """Test function"""
        with tf.Session() as sess:
            # Load the best model
            latest_checkpoint = tf.train.latest_checkpoint(self.config.save_dir)

            if tf.train.latest_checkpoint(self.config.save_dir) is not None:
                print("Restoring from {}...".format(
                    self.config.save_dir))
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                )

            # Test on the test data
            res = sess.run(
                fetches={
                    "seg_acc": self.seg_acc,
                },
                feed_dict={
                    self.seg_x: x_te,
                    self.seg_y: y_te,
                },
            )

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(
                res["seg_acc"]))


    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            
            pred_shape = [x.value for x in self.seg_logits.get_shape()]
            seg_shape = [x.value for x in self.seg_y.get_shape()]

            seg_preds = tf.reshape(self.seg_logits, [-1, pred_shape[3]])
            seg = tf.reshape(self.seg_y, [-1])

            # Create cross entropy loss for Segmentation
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=seg,
                    logits=seg_preds,
            ))

            # LSTM loss
            lstm_pred_shape = [x.value for x in self.lstm_out.get_shape()]
            lstm_out = tf.reshape(self.lstm_out, [-1, lstm_pred_shape[-1]])
            lstm_act = self.lstm_speed_y
            # lstm_act = tf.reshape(self.lstm_speed_y, [-1])

            self.loss += tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=lstm_act,
                    logits=tf.cast(lstm_act, tf.float32),  # TODO: This is broken!
                )
                # tf.nn.softmax_cross_entropy_with_logits(
                #     labels=lstm_act,
                #     logits=lstm_out,
                # )
            )

            # Create l2 regularizer loss and add
            l2_loss = tf.add_n([
                tf.nn.l2_loss(_v) for _v in tf.trainable_variables()]) # Note includes biases.
            self.loss += self.config.reg_lambda * l2_loss


            # Record summary for loss
            tf.summary.scalar("loss", self.loss)


def main(config):
    """The main function."""

    # Package data from directory into HD5 format
    if config.package_data:
        print("Packaging data into H5 format...")
        package_data(config.data_dir)
    else:
        print("Packaging data skipped.")

    # Load packaged data
    print("Loading data...")
    f = h5py.File('videoData.h5', 'r')
    data = []
    for group in f:
        """
        Keys of groups:
        ['class_colour',
         'class_id',
         'frame-10s',
         'info',
         'instance_colour',
         'instance_id',
         'raw_images',
         'video']
        """
        data.append(f[group])

    num_videos = len(data)

    # frame data and labels
    x = []
    y = []

    # contains previous frames
    lstm_x = []
    lstm_y = []

    # speed data and labels
    speed_x = []
    speed_y = []

    # iterate through videos
    for row in data:
        video = row.get('video')
        if not video: continue
        vector = row['info']
        assert video.shape[0] == vector.shape[0] 

        x.append(row['frame-10s'][:]) # Segmentation x, y
        y.append(row['class_id'][:])

        # current frame is the 30th so we want to also consider several previous ones
        batch = [video[29], video[28]]
        lstm_x.append(batch)
        lstm_y.append(row['frame-10s'][:])

        # motion data for lstm
        speed_batch = [vector[29], vector[28]]
        speed_x.append(speed_batch)
        speed_y.append(vector[30])


    # convert to np arrays
    x = np.asarray(x)
    y = np.asarray(y)
    lstm_x = np.asarray(lstm_x)
    lstm_y = np.asarray(lstm_y)
    speed_x = np.asarray(speed_x)
    speed_y = np.asarray(speed_y)

    print('Segmentation input shape: ', x.shape)
    print('LSTM X input shape: ', lstm_x.shape)
    print('Speed data input shape: ', speed_x.shape)
    
    assert len(x.shape) == 4, "Required: X is 4 tensor got %d." % len(x.shape)
    assert len(y.shape) == 4, "Required Y is 4 tensor got %d." % len(y.shape)
    assert len(lstm_x.shape) == 5, "Required: X is 5 tensor got %d." % len(lstm_x.shape)


    # Each label is pixel of [class_id, class_id, class_id], convert to single value
    y = y[:, :, :, 0]
    lstm_y = lstm_y[:, :, :, 0]
    
    def split(data):
        # 70% train, 20% val, 10% test split
        train_split = int(num_videos * 0.7)
        val_split = int(num_videos * 0.2) + train_split
        return data[:train_split], data[train_split:val_split], data[val_split:]
    
    # Seg data
    x_tr, x_va, x_te = split(x)
    y_tr, y_va, y_te = split(y)
    
    # LSTM Alexnet data
    lstm_x_tr, lstm_x_va, lstm_x_te = split(lstm_x)
    lstm_y_tr, lstm_y_va, lstm_y_te = split(lstm_y)
    
    # Motion data
    speed_y = np.apply_along_axis(_course_speed_labeler, -1, speed_y)
    speed_x_tr, speed_x_va, speed_x_te = split(speed_x)
    speed_y_tr, speed_y_va, speed_y_te = split(speed_y)
    
    seg_data = x_tr, y_tr, x_va, y_va
    lstm_data = lstm_x_tr, lstm_y_tr, lstm_x_va, lstm_y_va 
    speed_data = speed_x_tr, speed_y_tr, speed_x_va, speed_y_va

    # build network
    net = Network(x_tr.shape, lstm_x_tr.shape, config, speed_x_tr.shape)
    # train on train/val data
    net.train(seg_data, lstm_data, speed_data)
    
    # test on test data
    # net.test(x_te, y_te_labels)

def _course_speed_labeler(speed):
    if speed[0] < 1:
        return 1
    if abs(speed[1]) < 6:
        return 2
    if speed[1] < 0:
        return 3
    else:
        return 4

if __name__ == "__main__":

    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
