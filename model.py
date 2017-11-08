import tensorflow as tf

BATCH_SIZE = 1

dim_in = 128,128
dim_out = 32, 32


def _variable_with_weight_decay(name, shape, stddev, wd):
  var = tf.get_variable(name, shape,
      initializer=tf.truncated_normal_initializer(stddev=stddev,
        dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(inputs, filters, kernel_size, name):
  kernel = _variable_with_weight_decay(name,
      shape=[kernel_size[0], kernel_size[1], inputs.get_shape()[3], filters],
      stddev=5e-2,
      wd=0.0)
  conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.get_variable('biases_' + name, [filters],
      initializer=tf.constant_initializer(0.0), dtype=tf.float32)
  bias = tf.nn.bias_add(conv, biases)
  return bias

def model(data, differences, hue):
  with tf.variable_scope('conv1_1_data') as scope:
    conv1_1 = tf.nn.relu(conv2d(inputs=data, filters=256, 
        kernel_size=[3,3], name='conv1_1_data'))

  with tf.variable_scope('conv1_2_data') as scope:
    conv1_2 = tf.nn.relu(conv2d(inputs=conv1_1, filters=256,
        kernel_size=[5,5], name='conv1_2_data'))

  pool1_data = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
      padding='SAME',name='pool1_data')


  with tf.variable_scope('conv1_1_differences') as scope:
    conv1_1 = tf.nn.relu(conv2d(inputs=differences, filters=256,
        kernel_size=[3, 3], name='conv1_1_differences'))

  with tf.variable_scope('conv1_2_differences') as scope:
    conv1_2 = tf.nn.relu(conv2d(inputs=conv1_1, filters=256,
      kernel_size=[5, 5], name='conv1_2_differences'))


  pool1_differences = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1],
      strides=[1,2,2,1], padding='SAME', name='pool1_differences')


  with tf.variable_scope('conv1_1_hue') as scope:
    conv1_1 = tf.nn.relu(conv2d(inputs=hue, filters=256,
      kernel_size=[3, 3], name='conv1_1_hue'))

  with tf.variable_scope('conv1_2_hue') as scope:
    conv1_2 = tf.nn.relu(conv2d(inputs=conv1_1, filters=256,
      kernel_size=[5, 5], name='conv1_2_hue'))

  pool1_hue = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
      padding='SAME', name='pool1_hue')

  concat = tf.concat(axis=3, values=[pool1_data, pool1_differences, pool1_hue])


  with tf.variable_scope('conv2_1') as scope:
    conv2_1 = tf.nn.relu(conv2d(inputs=concat, filters=256,
      kernel_size=[3,3], name='conv2_1'))

  with tf.variable_scope('conv2_2') as scope:
    conv2_2 = tf.nn.relu(conv2d(inputs=conv2_1, filters=256,
      kernel_size=[5,5], name='conv2_2'))

  pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
      padding='SAME', name='pool2')

  with tf.variable_scope('conv3_1') as scope:
    conv3_1 = tf.nn.relu(conv2d(inputs=pool2, filters=256,
      kernel_size=[3,3], name='conv3_1'))

  with tf.variable_scope('conv3_2') as scope:
    conv3_2 = tf.nn.relu(conv2d(inputs=conv3_1, filters=256,
      kernel_size=[5,5], name='conv3_2'))

  with tf.variable_scope('conv3_3') as scope:
    conv3_3 = tf.nn.relu(conv2d(inputs=conv3_2, filters=64,
      kernel_size=[5,5], name='conv3_3'))

  with tf.variable_scope('fc1') as scope:
          reshape = tf.reshape(conv3_3, [BATCH_SIZE, -1])
          dim = reshape.get_shape()[1].value
          fc_dim = dim_out[0] * dim_out[1]
          weights = _variable_with_weight_decay('fc1',
              shape=[dim, fc_dim], stddev=5e-2, wd=0.004)
          biases = tf.get_variable('biases', [fc_dim],
              initializer=tf.constant_initializer(0.1), dtype=tf.float32)
          output = tf.nn.relu(tf.matmul(reshape, weights) + biases,
              name=scope.name)
  return output
