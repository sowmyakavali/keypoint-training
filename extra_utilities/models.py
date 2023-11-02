import tensorflow.keras.layers as tfl
import tensorflow as tf

def arch4(input_shape, output = 12):

    input_img = tf.keras.Input(shape=(input_shape, input_shape, 3))

    layer=tfl.Conv2D(filters= 38 , kernel_size= 5,strides=(2, 2),padding='same')(input_img)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    layer=tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)

    layer=tfl.Conv2D(filters= 114 , kernel_size= 3 ,strides=(2, 2),padding='same')(layer)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    
    layer=tfl.Conv2D(filters= 196 , kernel_size= 3 ,strides=(2, 2), padding='same')(layer)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)

    layer=tfl.Flatten()(layer)

    layer=tfl.Dense(units=500, activation='relu')(layer)
    layer=tfl.Dropout(0.2)(layer)

    outputs=tfl.Dense(units= output , activation='linear')(layer)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def arch6(input_shape, output = 12):

    input_img = tf.keras.Input(shape=(input_shape, input_shape, 3))
    
    layer=tfl.Conv2D(filters= 38 , kernel_size= 5,strides=(2, 2),padding='same')(input_img)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    layer=tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)

    layer=tfl.DepthwiseConv2D(kernel_size= 3 ,strides=(2, 2),padding='same')(layer)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    
    layer=tfl.Conv2D(filters= 196 , kernel_size= 3,strides=(2, 2),padding='same')(layer)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    
    layer=tfl.Flatten()(layer)

    layer=tfl.Dense(units=500, activation='relu')(layer)
    layer=tfl.Dropout(0.2)(layer)

    outputs=tfl.Dense(units = output , activation='linear')(layer)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def arch7():
    # from tensorflow.keras.layers.advanced_activations import LeakyReLU
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, MaxPool2D, ZeroPadding2D
    model = Sequential()

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96, 96, 3)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(12))
    # model.summary()
    return model