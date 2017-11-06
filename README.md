Keras is an object-oriented API for defining and training neural networks.

This module contains a pure-TensorFlow implementation of the Keras API,
allowing for deep integration with TensorFlow functionality.

See [keras.io](https://keras.io) for complete documentation and user guides.

1. checkpoint file from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
2. upzip file downloaded in step 1 and get vgg_16.ckpt inside the unzipped folder
3. put vgg_16.ckpt on /data_images directory
4. go to https://www.kaggle.com/c/cs5242-project-1/data and download train.csv, test_images.zip
   train_images.zip. unzip test_images.zip to data_images/test_images directory
   and train_images.zip to data_images/train_images directory. change train.csv to
   labels.csv and replace the labels.csv in /data_images/train_images.
   If there are existing images in these two folders, please remove the original images. 
4. Make sure folder structure is like in folder_structure_image
5. running train.py parameters can be configured in script parameters.
6. run train.py in background, you can check running status from generated file status.txt.
7. after run is finished, results will be save to submission.csv. 