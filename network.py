import os, h5py

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import get_config, print_usage
from util.preprocessing import package_data
from layerutils import fcl, convl


class Network:
    def __init__(self, x_shp, config):

        self.config = config

        # Get shape
        self.x_shp = x_shp

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
#        self._load_initial_weights()
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
        # Create Placeholders for inputs
        self.x_in = tf.placeholder(tf.float32, shape=x_in_shp, name="X_in")
        self.y_in = tf.placeholder(tf.int64, shape=x_in_shp[:-1], name="Y_in")
        self.batch_size = tf.shape(self.x_in)[0]

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):
            # we will make `n_mean`, `n_range`, `n_mean_in` and
            # `n_range_in` as scalar this time! This is how we often use in
            # CNNs, as we KNOW that these are image pixels, and all pixels
            # should be treated equally!

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

            # Compute the accuracy of the model. When comparing labels
            # elemwise, use tf.equal instead of `==`. `==` will evaluate if
            # your Ops are identical Ops.#
            self.pred = tf.argmax(self.logits, axis=3) # Argmax per pixel
            self.acc = tf.reduce_mean( 
                tf.to_float(tf.equal(self.pred, self.y_in))
            )

            # Record summary for accuracy
            tf.summary.scalar("accuracy", self.acc)

            # We also want to save best validation accuracy. So we do
            # something similar to what we did before with n_mean. Note that
            # these will also be a scalar variable
            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable(
                "best_va_acc", shape=(), trainable=False)
            #Assign op to store this value to TF variable
            self.acc_assign_op = tf.assign(self.best_va_acc, self.best_va_acc_in)

    def _load_initial_weights(self, sess):
        """Load weights from a file into network.
        Weights taken from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        It is a dict of lists """

        print("Loading pretrained weights for Alexnet...")
        # load weights from the file
        weights_dict = np.load(self.config.weights_dir, encoding='bytes').item()
        # Loop over all layer names stored in the weights dict
        #dict_keys(['fc6', 'fc7', 'fc8', 'conv3', 'conv2', 'conv1', 'conv5', 'conv4'])
        for op_name in weights_dict:

            # can define skips layer that will be trained fromscratch like this:
            # if op_name not in self.SKIP_LAYER:
            with tf.variable_scope("Network", reuse=True):

                with tf.variable_scope(op_name, reuse=True):
    
                    # Assign weights/biases to their corresponding tf variable
                    try:
                        weights, biases = weights_dict[op_name]

                    except:
                        import IPython; IPython.embed()
                        # Biases
                        var = tf.get_variable('biases', trainable=False)
                        sess.run(var.assign(biases))

                        # Weights
                        var = tf.get_variable('weights', trainable=False)
                        sess.run(var.assign(weights))
                            
        print("Weights loaded.")

###################
###
###    AlexNext implementation based on: https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/alexnet.py
###

    def alexNet(self):
        print("Building Alexnet into the network...")
        
        # Normalize using the above training-time statistics
        cur_in = (self.x_in - self.n_mean) / self.n_range
        print("Starting shape...", cur_in.shape)

        # 1st Layer Conv1
        cur_in = convl(cur_in, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        cur_in = tf.contrib.layers.max_pool2d(cur_in,
                                            [3, 3], 2, padding='VALID')

        # 2nd Layer Conv2
        cur_in = convl(cur_in, 5, 5, 256, 1, 1, groups=2, name='conv2')
        cur_in = tf.contrib.layers.max_pool2d(cur_in,
                                              [3, 3], 2, padding='VALID')

        # 3rd Layer Conv3
        cur_in = convl(cur_in, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer Conv4
        cur_in = convl(cur_in, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer Conv5
        cur_in = convl(cur_in, 3, 3, 256, 1, 1, groups=2, name='conv5')

#        # TODO: Decide to use this layer
#        cur_in = tf.contrib.layers.max_pool2d(cur_in,
#                                              [3, 3], 2, padding='VALID')
        
        cur_in = tf.contrib.layers.dropout(cur_in,
                                    0.3, is_training=True)
        # Fully connected layers with conv
        cur_in = convl(cur_in, 6, 6, 4096, 1, 1,  padding='VALID', name='fc6')

        cur_in = tf.contrib.layers.dropout(cur_in,
                                    0.3, is_training=True)

        # 8th Layer: FC (w ReLu) -> Dropout (as conv layer)
        cur_in = convl(cur_in, 1, 1, 4096, 1, 1,  padding='VALID', name='fc7')

        cur_in = tf.contrib.layers.dropout(cur_in,
                                    0.3, is_training=True)
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
            yshps = [x.value for x in self.y_in.get_shape()]

            cur_in = self.alexNet()
            print("Shape After alexnet..")
            print(cur_in.shape)
            
            self.logits = tf.contrib.layers.conv2d(cur_in, self.config.num_class, [1, 1],
                                   activation_fn=None,
                                   padding="VALID",
                                   biases_initializer=None)


            # Upscale logits to NWHC using nearest neighbor interpolation
            # turns (?,?,?, num_class) back to class scores for each pixel
            self.logits = tf.image.resize_images(
                images=self.logits,
                size=[yshps[1], yshps[2]], 
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            
            print("Shape After Reshape..", self.logits.shape)

            self.kernels_list = [
                _v for _v in tf.trainable_variables() if "kernel" in _v.name]         


    def train(self, x_tr, y_tr, x_va, y_va):
        """Training function.

        Parameters
        ----------
        x_tr : ndarray
            Training data.

        y_tr : ndarray
            Training labels.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """

        # ----------------------------------------
        # Preprocess data

        # We will simply use the data_mean for x_tr_mean, and 128 for the range
        # as we are dealing with image and CNNs now
        x_tr_mean = x_tr.mean()
        x_tr_range = 128.0

        # Report data statistic
        print("Training data before: mean {}, std {}, min {}, max {}".format(
            x_tr_mean, x_tr.std(), x_tr.min(), x_tr.max()
        ))

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            
            self._load_initial_weights(sess)
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

                # Get a random training batch. Notice that we are now going to
                # forget about the `epoch` thing. Theoretically, they should do
                # almost the same.
                ind_cur = np.random.choice(
                    len(x_tr), batch_size, replace=True)
                x_b = np.array([x_tr[_i] for _i in ind_cur])
                y_b = np.array([y_tr[_i] for _i in ind_cur])

                # Write summary every N iterations as well as the first
                # iteration. Use `self.config.report_freq`.
                K = self.config.report_freq
                # records 0 % K or step=1
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
                        "summary": self.summary_op,
                        "global_step": self.global_step,
                    }

                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.x_in: x_b,
                        self.y_in: y_b,
                    },
                )

               # Write Training Summary if we fetched it (no meta graph)
                if "summary" in res:
                   self.summary_tr.add_summary(
                       res["summary"], global_step=res["global_step"],
                   )
                   self.summary_tr.flush()

                   # Also save current model to resume when we write the
                   # summary.
                   self.saver_cur.save(
                       sess, self.save_file_cur,
                       global_step=self.global_step,
                       write_meta_graph=False,
                   )

                # Validate every N iterations and at the first iteration.
                V = self.config.val_freq
                b_validate = step % V == 0 and step!=0 or step == 1
                if b_validate:
                    res = sess.run(
                        fetches={
                           "acc": self.acc,
                           "summary": self.summary_op,
                           "global_step": self.global_step,
                        },
                        feed_dict={
                           self.x_in: x_va,
                           self.y_in: y_va,
                        })
                    # Write Validation Summary
                    self.summary_va.add_summary(
                       res["summary"], global_step=res["global_step"],
                    )
                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and
                    # best accuracy. We will only return the best W and b
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
                    "acc": self.acc,
                },
                feed_dict={
                    self.x_in: x_te,
                    self.y_in: y_te,
                },
            )

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(
                res["acc"]))


    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            pred_shape = [x.value for x in self.logits.get_shape()]
            seg_shape = [x.value for x in self.y_in.get_shape()]

            preds = tf.reshape(self.logits, [-1, pred_shape[3]])
            seg = tf.reshape(self.y_in, [-1])

            # Create cross entropy loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=seg,
                    logits=preds,
            ))

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

    nrows = len(data)

    x = []
    y = []
    for row in data:
        x.append(row['frame-10s'][:])
        y.append(row['class_id'][:])

    x = np.asarray(x)
    y = np.asarray(y)
    
    x_tr = x[:nrows//2]
    x_va = x[nrows//2:]

    y_tr = y[:nrows//2]
    y_va = y[nrows//2:]
    
    y_tr_labels = y[:, :, :, 0] # Each label is pixel of [class_id, class_id, class_id]
    y_va_labels = y[:, :, :, 0]
    
    net = Network(x_tr.shape, config)
    net.train(x_tr, y_tr_labels, x_va, y_va_labels)


if __name__ == "__main__":

    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
