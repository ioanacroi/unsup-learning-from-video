import tensorflow as tf
import scipy.io as scio
import os
#import matplotlib.pyplot as plt
import re
import numpy as np
import random

from tensorflow.python import debug as tf_debug


BATCH_SIZE = 16
NUM_EPOCHS = 10000

dim_in = 256, 256
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
  image = tf.image.resize_images(image, [310, 310])


  label = decode_raw(features['image/softseg'])

  label = tf.reshape(label, [256, 256, 1])
  label = tf.image.resize_images(label, [310, 310])

  img_for_crop = tf.concat([image, label], axis = 2)
  img_crop = tf.random_crop(img_for_crop, [256, 256, 4])

  r, g, b, la = tf.split(axis=2, num_or_size_splits=4, value=img_crop)
  image = tf.concat([r,g,b], axis = 2)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  label = la

  filename = features['image/filename']

  gray_img = tf.image.rgb_to_grayscale(image)
  ix = tf.py_func(compute_gradient_x, [gray_img], tf.float32)
  ix = tf.reshape(ix, [256, 256,1])
  iy = tf.py_func(compute_gradient_y, [gray_img], tf.float32)
  iy = tf.reshape(iy, [256, 256,1])

  differences = tf.concat([ix, iy], axis=2)

  hue = tf.image.rgb_to_hsv(image)
  hue, _, _ = tf.split(axis=2, num_or_size_splits=3, value=hue)

  return image, differences, hue, label, filename, width


# Input pipeline for batching
def inputs(tfrecords_filename, batch_size, num_epochs=None, evaluation=False):
    print tfrecords_filename

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                tfrecords_filename, num_epochs=num_epochs, shuffle=True)

        example, differences, hue, label, index, folder = read_and_decode(filename_queue)
        example_batch, differences_batch, hue_batch, label_batch, index_batch, folder_batch = tf.train.shuffle_batch(
            [example, differences, hue, label, index, folder], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

    #tf.summary.image('labels', label_batch, 5)

    label_batch = tf.cast(label_batch, tf.float32)
    return example_batch, differences_batch, hue_batch, label_batch, index_batch, folder_batch


def model(data, differences, hue, train=True):
	endpoints = {}


	with tf.variable_scope('down0'):
		#down0
		with tf.variable_scope('conv1_1'):
			down1 = tf.layers.conv2d(inputs=data, filters = 32, kernel_size=[3,3], padding = "same", name="conv1_1")
			down1 = tf.layers.batch_normalization(down1, training=train)
			down1 = tf.nn.relu(down1)	
			endpoints['conv1_1'] = down1

		with tf.variable_scope('conv1_2'):
			down1 = tf.layers.conv2d(inputs=down1, filters = 32, kernel_size=[3,3], padding = "same", name="conv1_2")
			down1 = tf.layers.batch_normalization(down1, training=train)
			down1 = tf.nn.relu(down1)
			endpoints['conv1_2'] = down1

		pool1 = tf.layers.max_pooling2d(down1, [2,2], 2, name="pool1")
		endpoints["pool1"] = pool1

	#down1
	with tf.variable_scope('down1'):
		with tf.variable_scope('conv2_1'):
			conv1 = tf.layers.conv2d(inputs=pool1, filters = 64, kernel_size=[3,3], padding = "same", name="conv2_1")
			conv1 = tf.layers.batch_normalization(conv1, training=train)
			conv1 = tf.nn.relu(conv1)	
			endpoints['conv2_1'] = conv1

		with tf.variable_scope('conv2_2'):
			conv2 = tf.layers.conv2d(inputs=conv1, filters = 64, kernel_size=[3,3], padding = "same", name="conv2_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['conv2_2'] = conv2

		pool1 = tf.layers.max_pooling2d(conv2, [2,2], 2, name="pool2")
		endpoints["pool2"] = pool1

	with tf.variable_scope('down2'):
		with tf.variable_scope('conv3_1'):
			#down2
			conv1 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size=[3,3], padding = "same", name="conv3_1")
			conv1 = tf.layers.batch_normalization(conv1, training=train)
			conv1 = tf.nn.relu(conv1)	
			endpoints['conv3_1'] = conv1

		with tf.variable_scope('conv3_2'):
			conv2 = tf.layers.conv2d(inputs=conv1, filters = 128, kernel_size=[3,3], padding = "same", name="conv3_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['conv3_2'] = conv2

		pool1 = tf.layers.max_pooling2d(conv2, [2,2], 2, name="pool3")
		endpoints["pool3"] = pool1

	with tf.variable_scope('down3'):
		with tf.variable_scope('conv4_1'):
			#down3
			conv1 = tf.layers.conv2d(inputs=pool1, filters = 256, kernel_size=[3,3], padding = "same", name="conv4_1")
			conv1 = tf.layers.batch_normalization(conv1, training=train)
			conv1 = tf.nn.relu(conv1)	
			endpoints['conv4_1'] = conv1

		with tf.variable_scope('conv4_2'):
			conv2 = tf.layers.conv2d(inputs=conv1, filters = 256, kernel_size=[3,3], padding = "same", name="conv4_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['conv4_2'] = conv2

		pool1 = tf.layers.max_pooling2d(conv2, [2,2], 2, name="pool4")
		endpoints["pool4"] = pool1

	with tf.variable_scope('down4'):
		with tf.variable_scope('conv5_1'):
			#down4
			conv1 = tf.layers.conv2d(inputs=pool1, filters = 512, kernel_size=[3,3], padding = "same", name="conv5_1")
			conv1 = tf.layers.batch_normalization(conv1, training=train)
			conv1 = tf.nn.relu(conv1)	
			endpoints['conv5_1'] = conv1

		with tf.variable_scope('conv5_2'):
			conv2 = tf.layers.conv2d(inputs=conv1, filters = 512, kernel_size=[3,3], padding = "same", name="conv5_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['conv5_2'] = conv2

		pool1 = tf.layers.max_pooling2d(conv2, [2,2], 2, name="pool5")
		endpoints["pool5"] = pool1

        with tf.variable_scope('center'):
		#center
		with tf.variable_scope('conv6_1'):
			conv1 = tf.layers.conv2d(inputs=pool1, filters = 1024, kernel_size=[3,3], padding = "same", name="conv6_1")
			conv1 = tf.layers.batch_normalization(conv1, training=train)
			conv1 = tf.nn.relu(conv1)	
			endpoints['conv6_1'] = conv1

		with tf.variable_scope('conv6_2'):
			conv2 = tf.layers.conv2d(inputs=conv1, filters = 1024, kernel_size=[3,3], padding = "same", name="conv6_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['conv6_2'] = conv2



	with tf.variable_scope('up4'):
		#up4
		up4 = tf.image.resize_images(endpoints['conv6_2'], [16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		up4 = tf.concat([endpoints['conv5_2'], up4], axis=3)

		with tf.variable_scope('up1_1'):
			conv2 = tf.layers.conv2d(inputs=up4, filters = 512, kernel_size=[3,3], padding = "same", name="up1_1")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up1_1'] = conv2

		with tf.variable_scope('up1_2'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 512, kernel_size=[3,3], padding = "same", name="up1_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up1_2'] = conv2

		with tf.variable_scope('up1_3'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 512, kernel_size=[3,3], padding = "same", name="up1_3")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up1_3'] = conv2

	with tf.variable_scope('up3'):
		#up3
		up4 = tf.image.resize_images(conv2, [32, 32], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		up4 = tf.concat([endpoints['conv4_2'], up4], axis=3)

		with tf.variable_scope('up2_1'):
			conv2 = tf.layers.conv2d(inputs=up4, filters = 256, kernel_size=[3,3], padding = "same", name="up2_1")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up2_1'] = conv2

		with tf.variable_scope('up2_2'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 256, kernel_size=[3,3], padding = "same", name="up2_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up2_2'] = conv2

		with tf.variable_scope('up2_3'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 256, kernel_size=[3,3], padding = "same", name="up2_3")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up2_3'] = conv2

	with tf.variable_scope('up2'):
		#up2
		up4 = tf.image.resize_images(conv2, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		up4 = tf.concat([endpoints['conv3_2'], up4], axis=3)

		with tf.variable_scope('up3_1'):
			conv2 = tf.layers.conv2d(inputs=up4, filters = 128, kernel_size=[3,3], padding = "same", name="up3_1")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up3_1'] = conv2

		with tf.variable_scope('up3_2'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 128, kernel_size=[3,3], padding = "same", name="up3_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up3_2'] = conv2

		with tf.variable_scope('up3_3'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 128, kernel_size=[3,3], padding = "same", name="up3_3")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up3_3'] = conv2

	with tf.variable_scope('up1'):
		#up1
		up4 = tf.image.resize_images(conv2, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		up4 = tf.concat([endpoints['conv2_2'], up4], axis=3)

		with tf.variable_scope('up4_1'):
			conv2 = tf.layers.conv2d(inputs=up4, filters = 64, kernel_size=[3,3], padding = "same", name="up4_1")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up4_1'] = conv2

		with tf.variable_scope('up4_2'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 64, kernel_size=[3,3], padding = "same", name="up4_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up4_2'] = conv2

		with tf.variable_scope('up4_3'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 64, kernel_size=[3,3], padding = "same", name="up4_3")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up4_3'] = conv2

	with tf.variable_scope('up0'):
		#up0
		up4 = tf.image.resize_images(conv2, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		up4 = tf.concat([endpoints['conv1_2'], up4], axis=3)

		with tf.variable_scope('up5_1'):
			conv2 = tf.layers.conv2d(inputs=up4, filters = 32, kernel_size=[3,3], padding = "same", name="up5_1")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up5_1'] = conv2

		with tf.variable_scope('up5_2'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 32, kernel_size=[3,3], padding = "same", name="up5_2")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up5_2'] = conv2

		with tf.variable_scope('up5_3'):
			conv2 = tf.layers.conv2d(inputs=conv2, filters = 32, kernel_size=[3,3], padding = "same", name="up5_3")
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)
			endpoints['up5_3'] = conv2


	classify = tf.layers.conv2d(endpoints['up5_3'], filters=1, kernel_size=[1,1], padding="same", activation=tf.nn.relu)

	return classify, endpoints


def loss(logits, labels, weighted = False):
       
	#soft loss 
        labels = tf.reshape(labels, [BATCH_SIZE, 256*256], name="labels")
	logits = tf.reshape(logits, [BATCH_SIZE, 256*256], name = "logits")

        l1 = tf.reshape(logits, [BATCH_SIZE, 256, 256, 1])
        tf.summary.image("logits", l1, 10)

	l2 = tf.reshape(labels, [BATCH_SIZE, 256, 256, 1])
	tf.summary.image("labels", l2, 10)

	loss = tf.reduce_mean(tf.square(logits-labels), name="eucli")
        
        return loss

def train(total_loss, global_step):
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        #opt = tf.train.GradientDescentOptimizer(0.001)
        #opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum = 0.9)
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
        tfrecords_filename = ["./tfrecords/train-" + str(x).zfill(5) + "-of-00020" for x in range(20)]

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
        summary_writer = tf.summary.FileWriter("./summaries_unet/", sess.graph)
        summary_writer.flush()

        train_op = train(lo, global_step)

        for var in tf.trainable_variables():
          tf.summary.histogram(var.op.name, var)

        init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
 
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(start_step, NUM_EPOCHS + 1):
          _, loss_value = sess.run([train_op, lo])

          if step % 100 == 0:
            print(step, loss_value)
          if step % 1000 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
          if step % 1000 == 0 and step!=0:
            saver.save(sess, "./checkpoints/unet/model_unet.ckpt-" + str(step))
        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
  tf.app.run()
