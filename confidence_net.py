import tensorflow as tf
import numpy as np
import os
from scipy import misc
import model_confidence as model
from argparse import ArgumentParser
import sys
import random

def compute_gradient_x(x):
  return np.gradient(x, axis=0).astype(np.float32)

def compute_gradient_y(x):
  return np.gradient(x, axis=1).astype(np.float32)

def calc_gradients(gray_img):
  ix = tf.py_func(compute_gradient_x, [gray_img], tf.float32)
  ix = tf.reshape(ix, [1,128,128,1])

  iy = tf.py_func(compute_gradient_y, [gray_img], tf.float32)
  iy = tf.reshape(iy, [1,128,128,1])
  derivatives = tf.concat((ix, iy), axis=3)

  return derivatives

def main(args):
  with tf.Graph().as_default():
    input_image = tf.placeholder(tf.uint8, shape=(128,128,3))
    input_image_f = tf.image.convert_image_dtype(input_image, dtype=tf.float32)

    input_soft_seg = tf.placeholder(tf.uint8, shape=(128,128,1))
    input_soft_seg_f = tf.image.convert_image_dtype(input_soft_seg, dtype=tf.float32)

    input_hsv = tf.placeholder(tf.uint8, shape=(128,128,3))

    input_hsv = tf.image.rgb_to_hsv(input_image_f)

    output = model.model(input_image_f, input_soft_seg_f, input_hsv, evaluation=True)[0]

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, args[1])

    img_dir = "/path/to/rgb/"
    soft_dir = "/path/to/softlabels/"



    differ = 0.0

    fr_nom = 0
    dict_nou = {}


    dir_out = "./test.txt"

    for dir in os.listdir(soft_dir):

      fr_cur = 0


      frames_dirs = os.listdir(soft_dir + dir)
      for frame in frames_dirs:
        fr_nom += 1
        fr_cur += 1
        data_path = img_dir + dir + '/' + frame.split('.')[0] + '.jpg'
        soft_path = soft_dir + dir + '/' + frame.split('.')[0] + '.png'


        img_rgb = misc.imread(data_path)
        img_rgb = misc.imresize(img_rgb, (128, 128))

        img_soft = misc.imread(soft_path)
        img_soft = misc.imresize(img_soft, (128, 128))
        img_soft = np.expand_dims(img_soft, axis=2)

        out, inpt, inpt_so = sess.run((output,input_image_f, input_soft_seg_f), feed_dict={input_image: img_rgb, input_soft_seg: img_soft})

        scor_nou = out[0][0]
        
        name_save = dir + '/' + frame.split('.')[0] + '.png' + ';' + str(scor_nou)

        if scor_nou >= 80:
          with open(dir_out, 'a') as f:
            f.write(name_save+'\n')


    sess.close()

if __name__ == '__main__':
  main(sys.argv)
