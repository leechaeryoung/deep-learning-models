from keras.layers import Conv2D, Conv2DTranspose, Input, Add, BatchNormalization, LeakyReLU, Reshape, Flatten, Dense
from keras.models import Model
import numpy as np

def build_skip_autoencoder():
    input_img = Input(shape=(256, 256, 1))
    y = Conv2D(32, (3, 3), padding='same', strides=(2,2))(input_img)
    y = LeakyReLU()(y)
    y = Conv2D(64, (3, 3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    y1 = Conv2D(128, (3, 3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y1)
    y = Conv2D(256, (3, 3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    y2 = Conv2D(256, (3, 3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y2)
    y = Conv2D(512, (3, 3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2D(1024, (3, 3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)

    vol = y.shape
    x = Flatten()(y)
    latent = Dense(128, activation='relu')(x)

    def lrelu_bn(inputs):
        lrelu = LeakyReLU()(inputs)
        bn = BatchNormalization()(lrelu)
        return bn

    y = Dense(np.prod(vol[1:]), activation='relu')(latent)
    y = Reshape((vol[1], vol[2], vol[3]))(y)
    y = Conv2DTranspose(1024, (3,3), padding='same')(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(512, (3,3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(256, (3,3), padding='same', strides=(2,2))(y)
    y = Add()([y2, y])
    y = lrelu_bn(y)
    y = Conv2DTranspose(256, (3,3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, (3,3), padding='same', strides=(2,2))(y)
    y = Add()([y1, y])
    y = lrelu_bn(y)
    y = Conv2DTranspose(64, (3,3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(32, (3,3), padding='same', strides=(2,2))(y)
    y = LeakyReLU()(y)
    output = Conv2DTranspose(1, (3,3), activation='sigmoid', padding='same', strides=(2,2))(y)

    model = Model(input_img, output)
    return model
