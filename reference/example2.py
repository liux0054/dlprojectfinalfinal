from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(3,200,200),name = 'image_input')

#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

# Generate a model with all layers (with top)
vgg16 = VGG16(weights=None, include_top=True)

#Add a layer where input is the output of the  second last layer
x = Dense(131, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model
my_model = Model(input=vgg16.input, output=x)
my_model.summary()


