
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras

def create_model(input_shape, n_out):

    model = Sequential()
    model.add(keras.applications.inception_resnet_v2.InceptionResNetV2(weights = "imagenet",
                                                                        include_top=False,
                                                                        input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(1024,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(n_out))
    model.add(Activation('sigmoid'))
    return model
