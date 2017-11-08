import tensorflow as tf
import numpy as np
from scipy import misc
import model
from argparse import ArgumentParser

def compute_gradient_x(x):
  return np.gradient(x, axis=0).astype(np.float32)

def compute_gradient_y(x):
  return np.gradient(x, axis=1).astype(np.float32)

def get_model_inputs(input_image):
  image_float32 = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
  hue = tf.image.rgb_to_hsv(image_float32)
  hue, _, _ = tf.split(axis=2,num_or_size_splits=3, value=hue)
  hue = tf.expand_dims(hue, 0)

  gray_img = tf.image.rgb_to_grayscale(image_float32)
  ix = tf.py_func(compute_gradient_x, [gray_img], tf.float32)
  ix = tf.reshape(ix, [1,128,128,1])

  iy = tf.py_func(compute_gradient_y, [gray_img], tf.float32)
  iy = tf.reshape(iy, [1,128,128,1])
  derivatives = tf.concat((ix, iy), axis=3)

  images_s = tf.expand_dims(image_float32, 0)

  return images_s, derivatives, hue


def main(args):
  with tf.Graph().as_default():
    input_image = tf.placeholder(tf.uint8, shape=(128,128,3))

    images, derivatives, hue = get_model_inputs(input_image)
    output = model.model(images, derivatives, hue)

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    image = misc.imread(args.in_file)

    orig_dim = image.shape
    image = misc.imresize(image, (128,128))
    out = sess.run(output, feed_dict={input_image: image})

    out = np.reshape(out, [1, 32, 32])

    out = out[0, :, :]
    out[out>255] = 255

    out = misc.imresize(out, orig_dim)
    out = out.astype(np.uint8)

    misc.imsave(args.out_file, out)

    sess.close()

if __name__ == '__main__':
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--in_file", dest="in_file", default="./car.jpg")
  arg_parser.add_argument("--out_file", dest="out_file", default="./segmentation.png")
  arg_parser.add_argument("--model", dest="model", default="./weights.ckpt")
  args = arg_parser.parse_args()
  main(args)
