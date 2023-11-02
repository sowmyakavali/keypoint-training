import cv2 
import os
import numpy as np 
import pandas as pd 
from math import sin, cos, pi
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers, callbacks, utils, applications, optimizers

from preprocess import preprocess
from tensorflow.keras.regularizers import l2
from v3_model import create_model


# for gpu growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

def load_images(image_data, size):
        images = []
        for idx, sample in image_data.iterrows():
            image = np.array(sample['Image'].split(' '), dtype=int)
            image = np.reshape(image, (size, size, 1))
            images.append(image)
        images = np.array(images)/255.
        return images

def load_keypoints(keypoint_data, size):
        keypoint_data = keypoint_data.drop('Image',axis = 1)
        keypoint_features = []
        for idx, sample_keypoints in keypoint_data.iterrows():
            keypoint_features.append(sample_keypoints)
        keypoint_features = np.array(keypoint_features, dtype = 'float')
        return keypoint_features

class training():

    def __init__(self, trainImages, trainKeypoints, epochs, batch_size, test_images, test_keypoints, imagesize, noOfkeypoints, finalsavepath, intermediate_path):
        self.train_images = trainImages
        self.train_keypoints = trainKeypoints
        self.epochs = epochs
        self.batch_size = batch_size
        self.testdata = (test_images, test_keypoints)
        self.imagesize = imagesize
        self.noOfkeypoints = noOfkeypoints
        self.intermediate = intermediate_path
        self.finalPath = finalsavepath
        self.channels = 1

    def load_images(image_data):
        images = []
        for idx, sample in image_data.iterrows():
            image = np.array(sample['Image'].split(' '), dtype=int)
            image = np.reshape(image, (96, 96, 1))
            images.append(image)
        images = np.array(images)/255.
        return images


    def load_keypoints(keypoint_data):
        keypoint_data = keypoint_data.drop('Image',axis = 1)
        keypoint_features = []
        for idx, sample_keypoints in keypoint_data.iterrows():
            keypoint_features.append(sample_keypoints)
        keypoint_features = np.array(keypoint_features, dtype = 'float')
        return keypoint_features


    def get_modelO(self):
        model = Sequential()
        
        pretrained_model = applications.MobileNetV3Small(input_shape=(self.imagesize, self.imagesize, self.channels), include_top=False, weights='imagenet')
        # pretrained_model = applications.EfficientNetV2S(input_shape=(self.imagesize, self.imagesize, self.channels), include_top=False, weights='imagenet')
        # pretrained_model = applications.MobileNet(input_shape=(self.imagesize, self.imagesize, self.channels), include_top=True, weights='imagenet')
        pretrained_model.trainable = True
        model.add(layers.Convolution2D(1, (2, 2), padding='same', input_shape=(self.imagesize, self.imagesize, 1), kernel_regularizer=l2(0.001)))
        model.add(layers.LeakyReLU(alpha = 0.1))
        model.add(pretrained_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.noOfkeypoints * 2))
        model.summary()
        return model
    
    def get_model(self):
        NUM_KEYPOINTS = 9*2
        IMG_SIZE = self.imagesize
        # Load the pre-trained weights of MobileNetV2 and freeze the weights
        backbone = applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 1)
        )
        backbone.trainable = False

        inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))
        x = applications.mobilenet_v2.preprocess_input(inputs)
        x = backbone(x)
        x = layers.Dropout(0.3)(x)
        x = layers.SeparableConv2D(
            NUM_KEYPOINTS, kernel_size=5, strides=1, activation="relu"
        )(x)
        x = layers.SeparableConv2D(
            NUM_KEYPOINTS, kernel_size=3, strides=1, activation="sigmoid"
        )(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(NUM_KEYPOINTS, activation="relu")(x)
        model = Model(inputs, outputs, name="keypoint_detector")
        model.summary()

        return model

    def start(self):
        model = create_model () #self.get_modelO()
        es = callbacks.EarlyStopping(monitor='loss', 
                                patience=50, 
                                verbose=1, 
                                mode='min', 
                                baseline=None, 
                                restore_best_weights=True)

        rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            factor=0.5, 
                                            patience=5, 
                                            min_lr=1e-15, 
                                            mode='min', 
                                            verbose=1)
        optimizer = tf.keras.optimizers.Adam(  learning_rate=0.006,
                                                beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-07,)
        model.compile(optimizer = optimizer, 
                        loss='mean_squared_error', 
                        metrics=['mse'])


        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.intermediate,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_freq='epoch',
                                                        period=2,
                                                        verbose=1)
        
        history = model.fit(self.train_images, self.train_keypoints, shuffle = True,   
                                epochs=int(self.epochs), batch_size=self.batch_size, 
                                validation_data=self.testdata, callbacks=[es, rlp, cp_callback])

        # model.save(os.path.join(r"D:\ML-projects\tryon-training\eyewearData\FinalDataset_96x96", "final.h5"))
        # score = model.evaluate(X_val, y_val)[1]
        # print("Final Score  : ", score)
        model.save(self.finalPath)

        # fig, ax = plt.subplots(3, 1, figsize=(20, 10))
        # df = pd.DataFrame(history.history)
        # df[['mae']].plot(ax=ax[0])
        # df[['loss']].plot(ax=ax[1])
        # df[['acc']].plot(ax=ax[2])
        # ax[0].set_title('Model MAE', fontsize=12)
        # ax[1].set_title('Model Loss', fontsize=12)
        # ax[2].set_title('Model Acc', fontsize=12)
        # fig.suptitle('Model Metrics', fontsize=18)
        


if __name__=="__main__":
    from datetime import datetime

    imagesize = 98
    batchsize = 32
    noOfepochs = 1000
    noOfkeypoints = 9
    #input data path
    data_path = r"FinalDataset_98x98\grayFinalData.csv"
    mainpath = r"FinalDataset_98x98"
    weights = "EfficientNetV2S"

    time = str(datetime.now()).split(" ")[0]

    finalsavepath = os.path.join(mainpath, f"shoe_{batchsize}b_{imagesize}s_{noOfkeypoints}kps_{time}_{weights}_Gray.h5")
    intermediate_path = os.path.join(mainpath, f"shoe_{batchsize}b_{imagesize}s_{noOfkeypoints}_{weights}kps_intermediate_Gray.h5")

    data = pd.read_csv(data_path)
    count = int(len(data)*0.7)
    
    traindata = data[:count]
    trainimages, trainkps = preprocess(traindata, imagesize)
    print("Shape of train_images:", np.shape(trainimages))

    test_data = data[count:]
    test_images = load_images(test_data, imagesize)
    test_keypoints = load_keypoints(test_data, imagesize)
    print("Shape of test_images:", np.shape(test_images))

    # trainImages = np.array(images)/255.0
    # keypointsData = np.array(keypoint_features, dtype = 'float')

    # X_train, X_val, y_train, y_val = train_test_split(trainImages, keypointsData, test_size=0.30, random_state=None)
    # print("X_train, X_val", X_train.shape, X_val.shape)

    trainfunc = training(trainimages, trainkps, noOfepochs, batchsize, test_images, test_keypoints, imagesize, noOfkeypoints, finalsavepath, intermediate_path)

    trainfunc.start()
