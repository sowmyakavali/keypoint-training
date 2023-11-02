import os
import cv2 
import numpy as np 
import pandas as pd 
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, callbacks, applications


from sklearn.model_selection import train_test_split

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from tensorflow.keras.regularizers import l2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

from models import arch7

def alter_brightness(images, keypoints, size):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.5, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.8, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))

def add_noise(images, keypoints, size, channels):
    noisy_images = []
    for image in images:
        noisy_image = cv2.add(image, 0.009*np.random.randn(size, size, channels))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]
        noisy_images.append(noisy_image.reshape(size, size, channels))
    return noisy_images, keypoints

def saturate_image(images, keypoints, size, channels):
    sat_images = []
    for image in images:
        saturated = tf.image.adjust_saturation(image, 6)
        sat_images.append(saturated.reshape(size, size, channels))
    return sat_images, keypoints

def left_right_flip(images, keypoints, size):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([size-coor if idx%2==0 else coor for idx, coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints



class training():

    def __init__(self, trainImages, trainKeypoints, epochs, noOfkeypoint, batch_size, size, valdata, intermediate_path, final_path, channels):
        self.train_images = trainImages
        self.train_keypoints = trainKeypoints
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_data = valdata
        self.size = size
        self.intermediate = intermediate_path
        self.finalPath = final_path
        self.noOfkps = noOfkeypoint
        self.channels = channels

    def get_model(self):
        model = Sequential()
        
        # pretrained_model = applications.MobileNet(input_shape=(self.size, self.size, self.channels), include_top=False, weights='imagenet')
        # pretrained_model = applications.MobileNetV3Small(input_shape=(self.size, self.size, self.channels), include_top=False, weights='imagenet')
        # pretrained_model = applications.EfficientNetV2S(input_shape=(self.size, self.size, self.channels), include_top=False, weights='imagenet')
        pretrained_model = applications.MobileNet(input_shape=(self.size, self.size, self.channels), include_top=False, weights='imagenet')
        pretrained_model.trainable = True

        model.add(layers.Convolution2D(3, (2, 2), padding='same', input_shape=(self.size, self.size, self.channels), kernel_regularizer=l2(0.02)))
        model.add(layers.LeakyReLU(alpha = 0.1))
        model.add(pretrained_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(self.noOfkps*2))
        model.summary()
        return model
    
    def start(self):
        model = self.get_model() # model = load_model(r"D:\Hand_manual\RGB_wistkp_model3_18042023.h5")
        
        es = callbacks.EarlyStopping(monitor='loss', 
                                patience=20, 
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
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                        loss='mse', 
                        metrics=['acc'])


        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.intermediate,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_freq='epoch',
                                                        period=5,
                                                        verbose=1)
        
        model.fit(train_images, train_keypoints, batch_size=self.batch_size, 
                                epochs=int(self.epochs), validation_data=self.val_data, 
                                callbacks=[es, rlp, cp_callback], 
                                shuffle=True)
        
        score = model.evaluate(X_val, y_val)[1]
        print("Final Score  : ", score)

        model.save(self.finalPath.replace(".h5", f"{round(score, 2)}.h5"))



if __name__=="__main__":
    mainpath = r"D:\Datasets\Datasets1to8\FinalDataset_98x98"
    datapath = os.path.join(mainpath, "RGBFinalData.csv")
    imagesPath = os.path.join(mainpath, "RGB_Images")
    csvData = pd.read_csv(datapath)
    time = str(datetime.now()).split(" ")[0]

    RESIZE = 96
    channels = 3
    noOfepochs = 500
    batchsize = 16
    noOfkeypoint = 9
    weights = ""
    finalsavepath = os.path.join(mainpath, f"shoe_{batchsize}b_{RESIZE}s_{noOfkeypoint}kps_{time}_{weights}.h5")
    intermediate_path = os.path.join(mainpath, f"shoe_{batchsize}b_{RESIZE}s_{noOfkeypoint}_{weights}kps_intermediate.h5")

    images = []
    keypoint_features = []

    for idx, sample in csvData.iterrows():
        filename = sample['Image']
        filepath = os.path.join(imagesPath, filename)
        # print(filepath)
        if os.path.isfile(filepath):
            image = cv2.imread(filepath)
            image = np.reshape(image, (RESIZE, RESIZE, channels))
            images.append(image)
            # keypoints data
            sample_keypoints = sample.drop('Image')
            keypoint_features.append(sample_keypoints)

    trainImages = np.array(images)/255.0
    keypointsData = np.array(keypoint_features, dtype = 'float')

    X_train, X_val, y_train, y_val = train_test_split(trainImages, keypointsData, test_size=0.30, random_state=None)
    print("X_train, X_val", X_train.shape, X_val.shape)

    # Augment method 1
    images1, keypoints1 = alter_brightness(X_train, y_train, RESIZE)
    train_images = np.concatenate((X_train, images1))
    train_keypoints = np.concatenate((y_train, keypoints1))

    # Augment method 2
    images2 , keypoints2 = add_noise(X_train, y_train, RESIZE, channels)
    train_images = np.concatenate((train_images, images2))
    train_keypoints = np.concatenate((train_keypoints, keypoints2))


    # # Augment method 3
    images4, keypoints4 = saturate_image(X_train, y_train, RESIZE, channels)
    train_images = np.concatenate((train_images, images4))
    train_keypoints = np.concatenate((train_keypoints, keypoints4))

    # # Augment method 4
    images5, keypoints5 = left_right_flip(X_train, y_train, RESIZE)
    train_images = np.concatenate((train_images, images5))
    train_keypoints = np.concatenate((train_keypoints, keypoints5))

    print(f"Total Training Images : {len(train_images)}\n Total Training keypoints : {len(train_keypoints)}")
    print(f"Total val Images : {len(X_val)}\n Total val keypoints : {len(y_val)}")

    # Start training
    trainfunc = training(train_images, train_keypoints, noOfepochs, noOfkeypoint, batchsize, RESIZE, (X_val, y_val), intermediate_path, finalsavepath, channels)
    trainfunc.start()
