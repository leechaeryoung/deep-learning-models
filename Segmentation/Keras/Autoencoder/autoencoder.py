from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D

def build_autoencoder():
    input_img = Input(shape=(256, 256, 1))

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    output = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(input_img, output)
    return model
