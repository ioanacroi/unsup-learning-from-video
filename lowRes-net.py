import tensorflow as tf
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import random

from tensorflow.python import debug as tf_debug


BATCH_SIZE = 32
NUM_EPOCHS = 1000000

dim_in = 128, 128
def _variable_with_weight_decay(name, shape, stddev, wd, train=True):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype), train)
  if wd is not None and train==True:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_on_cpu(name, shape, initializer, train=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
   Variable Tensor
  """
  dtype = tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=train)
  return var


def decode_jpeg(image_buffer, channels=3, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope('decode_jpeg', scope, [image_buffer]):
      # Decode the string as an RGB JPEG.
      # Note that the resulting image contains an unknown height and width
      # that is set dynamically by decode_jpeg. In other words, the height
      # and width of image is unknown at compile-time.
      image = tf.image.decode_jpeg(image_buffer, channels=channels)

      # After this point, all image pixels reside in [0,1)
      # until the very end, when they're rescaled to (-1, 1).  The various
      # adjust_* ops all require this range for dtype float.

      if channels == 3:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

      return image

def decode_raw(image_buffer, scope=None):
    with tf.name_scope('decode_raw', scope, [image_buffer]):
        image = tf.decode_raw(image_buffer, tf.uint8)

    return image

def decode_png(image_buffer, channels=3, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope('decode_png', scope, [image_buffer]):
      # Decode the string as an RGB JPEG.
      # Note that the resulting image contains an unknown height and width
      # that is set dynamically by decode_jpeg. In other words, the height
      # and width of image is unknown at compile-time.
      image = tf.image.decode_png(image_buffer, channels=channels, name = scope)

      # After this point, all image pixels reside in [0,1)
      # until the very end, when they're rescaled to (-1, 1).  The various
      # adjust_* ops all require this range for dtype float.

      if channels == 3:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

      return image

def compute_gradient_x(x):
  return np.gradient(x, axis=0).astype(np.float32)

def compute_gradient_y(x):
  return np.gradient(x, axis=1).astype(np.float32)

# Read a single example from TfRecord
# Must return a liniarized image, otherwise the batching will hang
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image/height': tf.FixedLenFeature([], tf.int64),
          'image/width': tf.FixedLenFeature([], tf.int64),
          'image/filename': tf.FixedLenFeature([], tf.string),
          'image/frame0': tf.FixedLenFeature([], tf.string),
          'image/softseg': tf.FixedLenFeature([], tf.string),
      })

  height = tf.cast(features['image/height'], tf.int64)
  width = tf.cast(features['image/width'], tf.int64)
  image = decode_raw(features['image/frame0'])
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.reshape(image, [256, 256, 3])
  image = tf.image.resize_images(image, [160, 160])


  label = decode_raw(features['image/softseg'])

  label = tf.reshape(label, [256, 256, 1])
  label = tf.image.resize_images(label, [160, 160])

  img_for_crop = tf.concat([image, label], axis = 2)
  img_crop = tf.random_crop(img_for_crop, [128, 128, 4])

  r, g, b, la = tf.split(axis=2, num_or_size_splits=4, value=img_crop)
  image = tf.concat([r,g,b], axis = 2)
  label = tf.image.resize_images(la, [32, 32])

  filename = features['image/filename']

  gray_img = tf.image.rgb_to_grayscale(image)
  ix = tf.py_func(compute_gradient_x, [gray_img], tf.float32)
  ix = tf.reshape(ix, [128, 128,1])
  iy = tf.py_func(compute_gradient_y, [gray_img], tf.float32)
  iy = tf.reshape(iy, [128, 128,1])

  differences = tf.concat([ix, iy], axis=2)

  hue = tf.image.rgb_to_hsv(image)
  hue, _, _ = tf.split(axis=2, num_or_size_splits=3, value=hue)

  return image, differences, hue, label, filename, width


# Input pipeline for batching
def inputs(tfrecords_filename, batch_size, num_epochs=None, evaluation=False):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                tfrecords_filename, num_epochs=num_epochs, shuffle=True)

        example, differences, hue, label, index, folder = read_and_decode(filename_queue)
        example_batch, differences_batch, hue_batch, label_batch, index_batch, folder_batch = tf.train.shuffle_batch(
            [example, differences, hue, label, index, folder], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)


    label_batch = tf.cast(label_batch, tf.float32)
    return example_batch, differences_batch, hue_batch, label_batch, index_batch, folder_batch


def model(data, differences, hue, evaluation = False, weighted = False, train=True):
        if train:
            BATCH_SIZE = 32
        else:
            BATCH_SIZE = 1

        end_points = {}
        data = tf.reshape(data, [BATCH_SIZE, dim_in[0], dim_in[1], 3], name = "data")
        differences = tf.reshape(differences, [BATCH_SIZE, dim_in[0], dim_in[1], 2], name = "diff")
        with tf.variable_scope('conv1_1_data') as scope:
                kernel = _variable_with_weight_decay('conv1_1_data',
                        shape=[3, 3, 3, 256],
                        stddev=5e-2,
                        wd=0.0)
                conv_l1 = tf.nn.conv2d(data, kernel, [1, 1, 1, 1], padding='SAME')
                biases1 = _variable_on_cpu('biases_conv1_1_data', [256], tf.constant_initializer(0.0))
                bias1 = tf.nn.bias_add(conv_l1, biases1)
                conv1_1 = tf.nn.relu(bias1, name=scope.name)
                end_points['conv1_1_data'] = conv1_1

        with tf.variable_scope('conv1_2_data') as scope:
                kernel = _variable_with_weight_decay('conv1_2_data',
                        shape=[5, 5, 256, 256],
                        stddev=5e-2,
                        wd=0.0)
                conv_l1 = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases1 = _variable_on_cpu('biases_conv1_2_data', [256], tf.constant_initializer(0.0))
                bias1 = tf.nn.bias_add(conv_l1, biases1)
                conv1_2 = tf.nn.relu(bias1, name=scope.name)
                end_points['conv1_2_data'] = conv1_2

        pool1_data = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                        name='pool1_data')
        end_points['pool1_data'] = pool1_data



        with tf.variable_scope('conv1_1_differences') as scope:
                kernel = _variable_with_weight_decay('conv1_1_differences',
                        shape=[3, 3, 2, 256],
                        stddev=5e-2,
                        wd=0.0)
                conv_l1 = tf.nn.conv2d(differences, kernel, [1, 1, 1, 1], padding='SAME')
                biases1 = _variable_on_cpu('biases_conv1_1_differences', [256], tf.constant_initializer(0.0))
                bias1 = tf.nn.bias_add(conv_l1, biases1)
                conv1_1 = tf.nn.relu(bias1, name=scope.name)
                end_points['conv1_1_differences'] = conv1_1

        with tf.variable_scope('conv1_2_differences') as scope:
                kernel = _variable_with_weight_decay('conv1_2_differences',
                        shape=[5, 5, 256, 256],
                        stddev=5e-2,
                        wd=0.0)
                conv_l1 = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases1 = _variable_on_cpu('biases_conv1_2_differences', [256], tf.constant_initializer(0.0))
                bias1 = tf.nn.bias_add(conv_l1, biases1)
                conv1_2 = tf.nn.relu(bias1, name=scope.name)
                end_points['conv1_2_differences'] = conv1_2

        pool1_differences = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                        name='pool1_differences')
        end_points['pool1_differences'] = pool1_differences


        with tf.variable_scope('conv1_1_hue') as scope:
                kernel = _variable_with_weight_decay('conv1_1_hue',
                        shape=[3, 3, 1, 256],
                        stddev=5e-2,
                        wd=0.0)
                conv_l1 = tf.nn.conv2d(hue, kernel, [1, 1, 1, 1], padding='SAME')
                biases1 = _variable_on_cpu('biases_conv1_1_hue', [256], tf.constant_initializer(0.0))
                bias1 = tf.nn.bias_add(conv_l1, biases1)
                conv1_1 = tf.nn.relu(bias1, name=scope.name)
                end_points['conv1_1_hue'] = conv1_1

        with tf.variable_scope('conv1_2_hue') as scope:
                kernel = _variable_with_weight_decay('conv1_2_hue',
                        shape=[5, 5, 256, 256],
                        stddev=5e-2,
                        wd=0.0)
                conv_l1 = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases1 = _variable_on_cpu('biases_conv1_2_hue', [256], tf.constant_initializer(0.0))
                bias1 = tf.nn.bias_add(conv_l1, biases1)
                conv1_2 = tf.nn.relu(bias1, name=scope.name)
                end_points['conv1_2_hue'] = conv1_2

        pool1_hue = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                                        name='pool1_hue')
        end_points['pool1_hue'] = pool1_hue


        concatenare = tf.concat([pool1_data, pool1_differences, pool1_hue], axis=3)

        with tf.variable_scope('conv2_1') as scope:
                kernel = _variable_with_weight_decay('conv2_1',
                          shape=[3,3,768,256],
                          stddev=5e-2,
                          wd=0.0)
                conv_l2 = tf.nn.conv2d(concatenare, kernel, [1,1,1,1], padding='SAME')
                biases2 = _variable_on_cpu('biases_conv2_1', [256], tf.constant_initializer(0.0))
                bias2 = tf.nn.bias_add(conv_l2, biases2)
                conv2_1 = tf.nn.relu(bias2, name=scope.name)
                end_points['conv2_1'] = conv2_1

        with tf.variable_scope('conv2_2') as scope:
                kernel = _variable_with_weight_decay('conv2_2',
                          shape=[5,5,256,256],
                          stddev=5e-2,
                          wd=0.0)
                conv_l2 = tf.nn.conv2d(conv2_1, kernel, [1,1,1,1], padding='SAME')
                biases2 = _variable_on_cpu('biases_conv2_2', [256], tf.constant_initializer(0.0))
                bias2 = tf.nn.bias_add(conv_l2, biases2)
                conv2_2 = tf.nn.relu(bias2, name=scope.name)
                end_points['conv2_2'] = conv2_2
 

        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        end_points['pool2'] = pool2

        with tf.variable_scope('conv3_1') as scope:
                kernel = _variable_with_weight_decay('conv3_1',
                          shape=[3,3,256,256],
                          stddev=5e-2,
                          wd=0.0)
                conv_l2 = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
                biases2 = _variable_on_cpu('biases_conv3_1', [256], tf.constant_initializer(0.0))
                bias2 = tf.nn.bias_add(conv_l2, biases2)
                conv3_1 = tf.nn.relu(bias2, name=scope.name)
                end_points['conv3_1'] = conv3_1

        with tf.variable_scope('conv3_2') as scope:
                kernel = _variable_with_weight_decay('conv3_2',
                          shape=[5,5,256,256],
                          stddev=5e-2,
                          wd=0.0)
                conv_l2 = tf.nn.conv2d(conv3_1, kernel, [1,1,1,1], padding='SAME')
                biases2 = _variable_on_cpu('biases_conv3_2', [256], tf.constant_initializer(0.0))
                bias2 = tf.nn.bias_add(conv_l2, biases2)
                conv3_2 = tf.nn.relu(bias2, name=scope.name)
                end_points['conv3_2'] = conv3_2

        with tf.variable_scope('conv3_3') as scope:
                kernel = _variable_with_weight_decay('conv3_3',
                          shape=[5,5,256,64],
                          stddev=5e-2,
                          wd=0.0)
                conv_l2 = tf.nn.conv2d(conv3_2, kernel, [1,1,1,1], padding='SAME')
                biases2 = _variable_on_cpu('biases_conv3_3', [64], tf.constant_initializer(0.0))
                bias2 = tf.nn.bias_add(conv_l2, biases2)
                conv3_3 = tf.nn.relu(bias2, name=scope.name)
                end_points['conv3_3'] = conv3_3

        net = conv3_3
        with tf.variable_scope('fc1') as scope:
                reshape = tf.reshape(net, [BATCH_SIZE, -1])
                dim = reshape.get_shape()[1].value
                weights = _variable_with_weight_decay('fc1', shape=[dim, 1024], stddev=5e-2, wd=0.004)
                biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
                fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
                out_pred = fc1
                end_points['fc1'] = fc1

        return out_pred, end_points

def loss(logits, labels, weighted = False):
       
        labels = tf.reshape(labels, [BATCH_SIZE, 32*32], name="labels")
	logits = tf.reshape(logits, [BATCH_SIZE, 32*32], name = "logits")

        l1 = tf.reshape(logits, [BATCH_SIZE, 32, 32, 1])
        tf.summary.image("logits", l1, 10)

	l2 = tf.reshape(labels, [BATCH_SIZE, 32, 32, 1])
	tf.summary.image("labels", l2, 10)

	loss = tf.reduce_mean(tf.square(logits-labels), name="eucli")
        
        return loss

def train(total_loss, global_step):
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)

def _add_loss_summaries(total_loss):
    """Add summaries for losses in the model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(l.op.name +' (raw)', l)
      tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def main(args):
    with tf.Graph().as_default():
        tfrecords_filename = ["tfrecords/train-" + str(x).zfill(5) + "-of-00020" for x in range(20)]

        print("!!!! MAKE SURE YOU SET THE TFRECORDS PATH !!!!!")
        random.shuffle(tfrecords_filename)
        random.shuffle(tfrecords_filename)

        config = tf.ConfigProto()
        start_step = 0

        global_step = tf.Variable(start_step, trainable=False)
 
        images, differences, hue, labels, _, _ = inputs(tfrecords_filename, BATCH_SIZE, NUM_EPOCHS)

        logits, end_points = model(images, differences, hue)
        _activation_summaries(end_points)

        lo = loss(logits, labels, weighted = False)
        _add_loss_summaries(lo)

        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=100)


        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./summaries/", sess.graph)
        summary_writer.flush()


        train_op = train(lo, global_step)

        for var in tf.trainable_variables():
          tf.summary.histogram(var.op.name, var)

        init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
 
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if len(args) == 2:
          saver.restore(sess, args[1])
        print("params")
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])) 

        for step in range(start_step, NUM_EPOCHS + 1):
          _, loss_value = sess.run([train_op, lo])

          if step % 10 == 0:
            print(step, loss_value)
          if step % 1000 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
          if step % 50000 == 0 and step!=0:
            saver.save(sess, "./checkpoints/model.ckpt-" + str(step))
        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
  tf.app.run()
