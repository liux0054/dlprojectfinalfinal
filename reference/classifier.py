import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

from reference.normalize_data import load_all_images


def model_fn(features, labels, mode, params):

    predictions = tf.reshape(vgg16_model, [-1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"class": predictions})

    loss = tf.losses.softmax_cross_entropy(labels, predictions)

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float64), predictions)
    }

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,eval_metric_ops=eval_metric_ops)

height = 224
width = 224
LEARNING_RATE = 0.01
seed = 7

np.random.seed(seed)
X, Y = load_all_images("./data/train")
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, Y, test_size=0.33, random_state=seed)


vgg16_model = VGG16(include_top=True, weights='imagenet', classes=131)

feature_columns = [tf.feature_column.numeric_column("x", shape=[height, width, 3])]

my_train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x" : np.array(X_train_set)},
    y = np.array(y_train_set),
    shuffle=True
)

my_test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": np.array(X_test_set)},
    y = np.array(y_test_set),
    shuffle=False
)



# Set model params
model_params = {"learning_rate": LEARNING_RATE}

# Instantiate Estimator
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)




