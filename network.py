import os, h5py

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import get_config, print_usage
from util.preprocessing import package_data


class Network:
    def __init__(self, x_shp, config):

        self.config = config

        # Get shape
        self.x_shp = x_shp

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        """Build placeholders."""

        # Get shape for placeholder
        x_in_shp = (None, *self.x_shp[1:])
        # Create Placeholders for inputs
        self.x_in = tf.placeholder(tf.float32, shape=x_in_shp, name="X_in")
        self.y_in = tf.placeholder(tf.int64, shape=x_in_shp, name="Y_in")
        self.y_lab = tf.placeholder(tf.int64, shape=x_in_shp[:-1], name="Y_lab")
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
            
            self.pred = tf.image.resize_images(
                images=self.logits,
                size=[244, 244],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            self.pred = tf.argmax(self.pred, axis=3)
            self.acc = tf.reduce_mean(
                tf.to_float(tf.equal(self.pred, self.y_lab))
            )

            # Record summary for accuracy
            tf.summary.scalar("accuracy", self.acc)

            # We also want to save best validation accuracy. So we do
            # something similar to what we did before with n_mean. Note that
            # these will also be a scalar variable
            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable(
                "best_va_acc", shape=(), trainable=False)
            # TODO: Assign op to store this value to TF variable
            self.acc_assign_op = tf.assign(self.best_va_acc, self.best_va_acc_in)  
            
    def _build_model(self):
        """Build our MLP network."""

        # Initializer and activations
        if self.config.activ_type == "relu":
            activ = tf.nn.relu
            kernel_initializer = tf.keras.initializers.he_normal()
        elif self.config.activ_type == "tanh":
            activ = tf.nn.tanh
            kernel_initializer = tf.glorot_normal_initializer()

        # Build the network (use tf.layers)
        with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
            # Normalize using the above training-time statistics
            cur_in = (self.x_in - self.n_mean) / self.n_range

            num_unit = self.config.num_unit

            print("Input shape" , cur_in.shape)
            cur_in = tf.layers.conv2d(inputs=cur_in,
                                      filters=96,
                                      kernel_size=[11, 11],
                                      strides=[4, 4],
                                      kernel_initializer=kernel_initializer,
                                      padding='VALID',
                                      activation=tf.nn.relu)
            
            print("Layer1" , cur_in.shape)
            cur_in = tf.contrib.layers.max_pool2d(cur_in,
                                                [3, 3], 2         
            )
            print("Layer2" , cur_in.shape)

            cur_in = tf.layers.conv2d(cur_in, 256, [5, 5],
             activation=tf.nn.relu)
            print("Layer3" , cur_in.shape)

#            cur_in = tf.contrib.layers.max_pool2d(cur_in,
#                                                  [3, 3], 2)            
            print("Layer4" , cur_in.shape)

            cur_in = tf.layers.conv2d(cur_in, 384, [3, 3],
                                         activation=tf.nn.relu)
            print("Layer5" , cur_in.shape)

            cur_in = tf.layers.conv2d(cur_in, 384, [3, 3],
                                    activation=tf.nn.relu)
            print("Layer6" , cur_in.shape)

            cur_in = tf.layers.conv2d(cur_in, 256, [3, 3],
                                    activation=tf.nn.relu)
            print("Layer7" , cur_in.shape)

#            cur_in = tf.contrib.layers.max_pool2d(cur_in,
#                                                  [3, 3], 2)            
            
            print("Layer8" , cur_in.shape)

            cur_in = tf.layers.conv2d(cur_in, 4096, [5, 5],
                                   activation=tf.nn.relu)
            print("Layer9" , cur_in.shape)

            cur_in = tf.contrib.layers.dropout(cur_in,
                                        0.3, is_training=True)
            print("Layer10" , cur_in.shape)

            cur_in = tf.layers.conv2d(cur_in, 4096, [1, 1],
                                   activation=tf.nn.relu)
            print("Layer11" , cur_in.shape)

            cur_in = tf.contrib.layers.dropout(cur_in,
                                        0.3, is_training=True)
            print("Layer12" , cur_in.shape)

            self.logits = tf.contrib.layers.conv2d(cur_in, self.config.num_class, [1, 1],
                                   activation_fn=None,
                                   padding="VALID",
                                   biases_initializer=None)
            
            print("Logits last layer shape", self.logits.shape)
            
            yshps = [x.value for x in self.y_in.get_shape()]
#            self.logits = tf.image.resize_nearest_neighbor(self.logits,
#                                                   [640,
#                                                    360])
            print("Logits Final layer shape", self.logits.shape)
            
#            self.logits = upsample(self.logits, config.num_class, 32, "testa")
#            N, H, W, C = cur_in.shape
#            upsample_shape = tf.pack([N, 640, 360, C]) 
#            self.logits =        
##                self.logits = tf.layers.conv2d_transpose(cur_in)
            self.logits = tf.image.resize_images(
                images=self.logits,
#                size=[self.batch_size, 640*360],
                size=[244, 244],

                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
#            
            print("Logits shape: ", self.logits.shape)
            # Get list of all weights in this scope. They are called "kernel"
            # in tf.layers.dense.
            self.kernels_list = [
                _v for _v in tf.trainable_variables() if "kernel" in _v.name]    

    def train(self, x_tr, y_tr, x_va, y_va, y_lab):
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
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            # Assign normalization variables from statistics of the train data
            sess.run(self.n_assign_op, feed_dict={
                self.n_mean_in: x_tr_mean,
                self.n_range_in: x_tr_range,
            })

     
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
                    len(x_tr), batch_size, replace=False)
                x_b = np.array([x_tr[_i] for _i in ind_cur])
                y_b = np.array([y_tr[_i] for _i in ind_cur])
                y_lab_b = np.array([y_lab[_i] for _i in ind_cur])
                # TODO: Write summary every N iterations as well as the first
                # iteration. Use `self.config.report_freq`. Make sure that we
                # write at the first iteration, and every kN iterations where k
                # is an interger. HINT: we write the summary after we do the
                # optimization.
#                b_write_summary = (step % self.config.report_freq) == 0
#                if b_write_summary:
#                    fetches = {
#                        "optim": self.optim,
#                        "summary": self.summary_op,
#                        "global_step": self.global_step,
#                    }
#                else:
                fetches = {
                    "optim": self.optim,
                }

            
                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.x_in: x_b,
                        self.y_in: y_b,
                        self.y_lab: y_lab_b,
                    },
                )

                # Write Training Summary if we fetched it (don't write
                # meta graph). See that we actually don't need the above
                # `b_write_summary` actually :-). I know that we can check this
                # with b_write_summary, but let's check `res` to do this as an
                # exercise.
#                if "summary" in res:
#                    self.summary_tr.add_summary(
#                        res["summary"], global_step=res["global_step"],
#                    )
#                    self.summary_tr.flush()
#
#                    # Also save current model to resume when we write the
#                    # summary.
#                    self.saver_cur.save(
#                        sess, self.save_file_cur,
#                        global_step=self.global_step,
#                        write_meta_graph=False,
#                    )

#                 Validate every N iterations and at the first
#                 iteration. Use `self.config.val_freq`. Make sure that we
#                 validate at the correct iterations. HINT: should be similar
#                 to above.
#                b_validate = (step % self.config.report_freq) == 0
#                if b_validate:
#                    res = sess.run(
#                        fetches={
#                            "acc": self.acc,
#                            "summary": self.summary_op,
#                            "global_step": self.global_step,
#                        },
#                        feed_dict={
#                            self.x_in: x_va,
#                            self.y_in: y_va,
#                            self.y_lab: y_lab_b,
#
#                        })
#                    print("Accuracy", res['acc'])
##                    # Write Validation Summary
#                    self.summary_va.add_summary(
#                        res["summary"], global_step=res["global_step"],
#                    )
#                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and
                    # best accuracy. We will only return the best W and b
#                    if res["acc"] > best_acc:
#                        best_acc = res["acc"]
#                        # TODO: Write best acc to TF variable
#                        sess.run(self.acc_assign_op, feed_dict={
#                            self.best_va_acc_in: best_acc
#                        })

                        # Save the best model
#                        self.saver_best.save(
#                            sess, self.save_file_best,
#                            write_meta_graph=False,
#                        )
    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            pred_shape = [x.value for x in self.logits.get_shape()]
            seg_shape = [x.value for x in self.y_lab.get_shape()]
#            tf.reshape(ylab, [-1, ]
#            preds = tf.image.resize_nearest_neighbor(self.logits, [
#                                                    seg_shape[1],
#                                                    seg_shape[2]]
#                                                  )

            preds = tf.reshape(self.logits, [-1, pred_shape[3]])
            seg = tf.reshape(self.y_lab, [-1])
            
            # Create cross entropy loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=seg,
                    logits=preds,
            ))
#            print(self.logits.shape)
#            print(self.y_in.shape)
            # Create l2 regularizer loss and add
            l2_loss = tf.add_n([
                tf.reduce_sum(_v**2) for _v in self.kernels_list])
            self.loss += self.config.reg_lambda * l2_loss

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)            
    
    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "./train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "./valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")
        
        
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
    
    d = data[0]
    nrows = len(data)
    x_tr = data[:nrows//2]
    x_va = data[nrows//2:]
    
    x = []
    y = []
    for row in data:    
        x.append(row['frame-10s'][:])
       # y.append(row['class_colour'][:])
        y.append(row['instance_id'][:])

    
    x = np.asarray(x)
    y = np.asarray(y)

    x_tr = x[:nrows//2]
    x_va = x[nrows//2:]
    
    y_tr = y[:nrows//2]
    y_va = y[nrows//2:]
    
    # TODO: load from csv note pixels are backwards
    class2lab = {(0 ,0, 0): 0,
                 (136, 136, 136): 1,
                 (67, 67, 67): 0}
    
    l = lambda x: class2lab[tuple(x)] if tuple(x) in class2lab else 0
    y_tr_labels = np.apply_along_axis(l, -1, y_tr)
    
    net = Network(x_tr.shape, config)
    net.train(x_tr, y_tr, x_va, y_va, y_tr_labels)
    
##-------------------------------------------------------------------------------
## Author: Lukasz Janyst <lukasz@jany.st>
## Date:   14.06.2017
##
## This essentially is the code from
## http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
## with minor modifications done by me. See test_upscale.py to see how it works.
##-------------------------------------------------------------------------------
#
#import tensorflow as tf
#import numpy as np
#
#-------------------------------------------------------------------------------
def get_bilinear_filter(filter_shape, upscale_factor):
    """
    Creates a weight matrix that performs a bilinear interpolation
    :param filter_shape:   shape of the upscaling filter
    :param upscale_factor: scaling factor
    :return:               weight tensor
    """

    kernel_size = filter_shape[1]

    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location)/upscale_factor)) * \
                    (1 - abs((y - centre_location)/upscale_factor))
            bilinear[x, y] = value

    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights

#-------------------------------------------------------------------------------
def upsample(x, n_channels, upscale_factor, name):
    """
    Create an upsampling tensor
    :param x:              input tensor
    :param n_channels:     number of channels
    :param upscale_factor: scale factor
    :param name:           name of the tensor
    :return:               upsampling tensor
    """

    kernel_size = 2*upscale_factor - upscale_factor%2
    stride      = upscale_factor
    strides     = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_shape = tf.shape(x)

        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, n_channels]

        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        weights = get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv
    
if __name__ == "__main__":

    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


