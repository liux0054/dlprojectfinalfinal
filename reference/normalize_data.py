import tensorflow as tf
import os
import numpy as np
import csv

height = 224
width = 224

def normalize_image(image_name):
    image = tf.image.decode_jpeg(tf.read_file(image_name), channels=3)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    return image


def load_all_images(dir_path):
    num_of_files = len([name for name in os.listdir(dir_path)])
    images = list()
    y = list()
    for i in range(num_of_files-1):
        image = normalize_image(dir_path + '/' + str(i)+'.jpg')
        images.append(image)

    with open(dir_path + '/train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            y.append(row['category'])

    return images, y

