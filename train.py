"""
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release v1.2
Based on PyTorch example from Justin Johnson
(https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c)
Required packages: tensorflow (v1.2)
Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:
```
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
```
The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

import argparse
import os

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import random
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='./data/train_images', type=str)
parser.add_argument('--test_dir', default='./data/test_images', type=str)
parser.add_argument('--label_path', default='./data/train.csv', type=str)
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=50, type=int)
parser.add_argument('--num_epochs2', default=50, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--train_val_split', default=0.1, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    master = pd.read_csv(args.label_path)
    master.head()
    filenames = list()
    labels = list()
    for i in range(len(master)):
        filenames.append(os.path.join(directory, str(master.ix[i][0])))
        labels.append(master.ix[i][1])

    train_file_names = list()
    train_labels = list()
    val_file_names = list()
    val_labels = list()

    for i in range(len(labels)):
        if random.random() < args.train_val_split:
            val_file_names.append(filenames[i])
            val_labels.append(labels[i])
        else:
            train_file_names.append(filenames[i])
            train_labels.append(labels[i])

    return train_file_names, train_labels, val_file_names, val_labels


def list_test_images(directory):
    num_of_files = len([name for name in os.listdir(directory)])
    test_filenames = list()
    labels = list()
    for i in range(num_of_files):
        test_filenames.append(directory + '/' + str(i)+'.jpg')
        labels.append(100)
    return test_filenames, labels, num_of_files


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def get_prediction(sess, prediction, is_training, dataset_init_op, num_of_test_files):
    sess.run(dataset_init_op)

    with open('test.csv', "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        head = ['image_name', 'category']
        writer.writerow(head)
        csv_file.close()

    counter = 0
    while True:
        if counter == num_of_test_files:
            break
        try:
            if counter < num_of_test_files:
                pred = sess.run(prediction, {is_training: False})
                pred = pred.reshape(-1, 1)
                with open('test.csv', 'a') as submission_file:
                    writer = csv.writer(submission_file, delimiter=',')
                    for row in pred:
                        w_row = [str(counter)+'.jpg', str(row[0])]
                        writer.writerow(w_row)
                        counter += 1
                        if counter == num_of_test_files:
                            break
                    submission_file.close()
            else:
                break
        except tf.errors.OutOfRangeError:
            break
    return


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels, val_filenames, val_labels = list_images(args.train_dir)
    num_classes = len(set(train_labels+val_labels))

    test_filenames, dummy_lables, num_of_test_files = list_test_images(args.test_dir)
    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`

    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image so its smaller side is 256 pixels long
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            return resized_image, label

        def _parse_function_test(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            return resized_image, label

        # Preprocessing (for training)
        # (3) Take a random 224x224 crop to the scaled image
        # (4) Horizontally flip the image with probability 1/2
        # (5) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def training_preprocess(image, label):
            crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
            flip_image = tf.image.random_flip_left_right(crop_image)                # (4)
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = flip_image - means                                     # (5)
            rotated_image_1 = tf.image.rot90(centered_image, k=1)
            rotated_image_2 = tf.image.rot90(centered_image, k=3)
            prob = random.random()
            if prob < 0.25:
                if random.random() < 0.5:
                    return rotated_image_1, label
                else:
                    return rotated_image_2, label
            elif prob<0.5:
                distorted_image = tf.image.random_brightness(centered_image, max_delta=63)
                contrast_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
                return contrast_image, label
            else:
                return centered_image, label

        # Preprocessing (for validation)
        # (3) Take a central 224x224 crop to the scaled image
        # (4) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def val_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means                                     # (4)
            return centered_image, label

        def test_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means                                     # (4)
            return centered_image, label

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.concat([train_filenames, train_filenames],0),
                                                                              tf.concat([train_labels, train_labels], 0)))
        train_dataset = train_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess, num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # Validation dataset
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(val_preprocess,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)

        # test dataset
        test_filenames = tf.constant(test_filenames)
        dummy_lables = tf.constant(dummy_lables)
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_filenames, dummy_lables))
        test_dataset = test_dataset.map(_parse_function_test,
             num_threads=args.num_workers, output_buffer_size=args.batch_size)
        test_dataset = test_dataset.map(test_preprocess,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_test_dataset = test_dataset.batch(1)

        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)

        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)
        test_init_op = iterator.make_initializer(batched_test_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        vgg = slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        fc8_init = tf.variables_initializer(fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
        fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)

        tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        init_fn(sess)  # load the pretrained weights
        sess.run(fc8_init)  # initialize the new fc8 layer

        # Update only the last layer for a few epochs.
        for epoch in range(args.num_epochs1):
            # Run an epoch over the training data.
            with open("status.txt", "a") as myfile:
                myfile.write('Starting epoch %d / %d ' % (epoch + 1, args.num_epochs1) + '\n')
                myfile.close()
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(fc8_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            with open("status.txt", "a") as myfile:
                myfile.write('Train last layer epoch ' + str(epoch) + '\n')
                myfile.write('Train accuracy: %f \n' % train_acc)
                myfile.write('Val accuracy: %f \n' % val_acc)
                myfile.close()

        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs2):
            with open("status.txt", "a") as myfile:
                myfile.write('Starting epoch %d / %d ' % (epoch + 1, args.num_epochs2) + '\n')
                myfile.close()
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(full_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            with open("status.txt", "a") as myfile:
                myfile.write('Train whole model epoch ' + str(epoch) +'\n')
                myfile.write('Train accuracy: %f \n' % train_acc)
                myfile.write('Val accuracy: %f \n' % val_acc)
                myfile.close()

        get_prediction(sess, prediction, is_training, test_init_op, num_of_test_files)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)