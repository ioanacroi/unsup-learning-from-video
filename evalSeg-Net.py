import tensorflow as tf
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import re
import numpy as np

BATCH_SIZE = 16
NUM_EPOCHS = 1000000

dim_in = 128,128

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

      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

      return image

def decode_png(image_buffer, channels=1, scope=None):
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

      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

      return image

def compute_gradient_x(x):
  return np.gradient(x, axis=0).astype(np.float32)

def compute_gradient_y(x):
  return np.gradient(x, axis=1).astype(np.float32)

# Read a single example from TfRecord
# Must return a liniarized image, otherwise the batching will hang

def calc_gradients(img):
  gray_img = tf.image.rgb_to_grayscale(img)
  ix = tf.py_func(compute_gradient_x, [gray_img], tf.float32)
  ix = tf.reshape(ix, [128, 128,1])
  iy = tf.py_func(compute_gradient_y, [gray_img], tf.float32)
  iy = tf.reshape(iy, [128, 128,1])

  differences = tf.concat((ix, iy), 2)

  return differences


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
          'image/label': tf.FixedLenFeature([], tf.float32),
      })

  height = tf.cast(features['image/height'], tf.int64)
  width = tf.cast(features['image/width'], tf.int64)
  image = decode_jpeg(features['image/frame0'])
  image = tf.image.resize_images(image, [128, 128])

  soft_seg = decode_png(features['image/softseg'])
  soft_seg = tf.image.resize_images(soft_seg, [128, 128])

  filename = features['image/filename']
  
  label = tf.cast(features['image/label'], tf.float32)

  hsv = tf.image.rgb_to_hsv(image)

  return image, soft_seg, hsv, label, filename, width


def _variable_with_weight_decay(name, shape, stddev, wd, train=False):
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

def _variable_on_cpu(name, shape, initializer, train=False):
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

# Input pipeline for batching
def inputs(tfrecords_filename, batch_size, num_epochs=None, evaluation=False):
    print tfrecords_filename

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                tfrecords_filename, num_epochs=num_epochs, shuffle=True)


        example, soft_seg, hue, label, index, folder = read_and_decode(filename_queue)
        example_batch, soft_seg_batch, hue_batch, label_batch, index_batch, folder_batch = tf.train.shuffle_batch(
            [example, soft_seg, hue, label, index, folder], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

    label_batch = tf.cast(label_batch, tf.float32)
    return example_batch, soft_seg_batch, hue_batch, label_batch, index_batch, folder_batch

def branch(inp, name):
  endpoints={}
  with tf.variable_scope('conv1_1_' + name):
    conv1_1 = tf.layers.conv2d(inputs=inp, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    endpoints['conv1_1_' + name] =conv1_1
  with tf.variable_scope('conv1_2_' + name):
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=256, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
    endpoints['conv1_2_' + name] =conv1_2

  pool = tf.layers.max_pooling2d(conv1_2, [2,2], 2, name="pool_"+name)
  endpoints['pool_' + name]=pool

  return endpoints


def model(data, soft_seg, hsv, evaluation = False, weighted = False):

      endpoints = {}
      if evaluation:
        global BATCH_SIZE
        BATCH_SIZE = 1

      data = tf.reshape(data, [BATCH_SIZE, dim_in[0], dim_in[1], 3], name = "data")
      soft_seg = tf.reshape(soft_seg, [BATCH_SIZE, dim_in[0], dim_in[1], 1], name = "soft_seg")
      hsv = tf.reshape(hsv, [BATCH_SIZE, dim_in[0], dim_in[1], 3], name = "hsv")

      branch_data = branch(data, 'data')
      branch_seg = branch(soft_seg, "seg")
      branch_hsv = branch(hsv, "hsv")

      endpoints.update(branch_data)
      endpoints.update(branch_seg)
      endpoints.update(branch_hsv)

      concatenare = tf.concat([endpoints['pool_data'], endpoints['pool_seg'], endpoints['pool_hsv']], 3)

      with tf.variable_scope('conv2_1'):
        conv = tf.layers.conv2d(inputs=concatenare, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        endpoints['conv2_1'] =conv

      with tf.variable_scope('conv2_2'):
        conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        endpoints['conv2_2'] =conv

      pool = tf.layers.max_pooling2d(conv, [2,2], 2, name="pool2")
      endpoints['pool2'] = pool

      with tf.variable_scope('conv3_1'):
        conv = tf.layers.conv2d(inputs=pool, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        endpoints['conv3_1'] = conv

      with tf.variable_scope('conv3_2'):
        conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        endpoints['conv3_2'] = conv

      with tf.variable_scope('conv3_3'):
        conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
        endpoints['conv3_3'] = conv


      with tf.variable_scope('fc1'):
        fc = tf.layers.dense(tf.reshape(conv, [BATCH_SIZE, -1]), 512, activation=tf.nn.relu)
        endpoints['fc1'] = fc

      with tf.variable_scope('fc2'):
        fc = tf.layers.dense(fc, 1)
        endpoints['fc2'] = fc

      out_pred = fc

      return out_pred, endpoints

def loss(logits, labels, weighted = False):
        logits = tf.reshape(logits, [BATCH_SIZE, 1], name = "logits")

        labels = tf.reshape(labels, [BATCH_SIZE, 1], name="labels")


        loss = tf.reduce_mean(tf.square(logits-labels), name="eucli")

        return loss

def train(total_loss, global_step):
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)

        train_op = opt.minimize(total_loss, global_step=global_step)
        return (train_op, global_step)


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

if __name__ == '__main__':
  tf.app.run()
