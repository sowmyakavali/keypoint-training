# Import common package
import os
import cv2
import math
import json
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from IPython.display import Image, display
from matplotlib import pyplot as plt
# Import Package for model building
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Add, Conv2D, UpSampling2D, Dropout
from tensorflow.keras import models, layers, optimizers, losses, metrics, datasets
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *



class DataPreparation:
    def __init__(self, hm_height, hm_width, ip_width, ip_height, sigma, stride, noOfkps):
        
        self.heatmap_width = hm_width
        self.heatmap_height = hm_height
        self.input_img_width = ip_width
        self.input_img_height = ip_height
        self.sigma = sigma
        self.stride = stride
        self.noOfkps = noOfkps

    def create_heatmap(self, kp):
        
        all_joints = [[round(x[0] * (self.heatmap_width / self.input_img_width)), 
                       round(x[1] * (self.input_img_height / self.heatmap_height))] for x in kp]
        
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width , self.noOfkps), 
                           dtype=np.float32)

        if len(all_joints) == self.noOfkps:
            for plane_idx, joint in enumerate(all_joints):
                if joint:
                    self.put_heatmap_on_plane(heatmap, plane_idx, joint)
        

        return heatmap


    def put_heatmap_on_plane(self, heatmap, plane_idx, joint):
        
        start = self.stride / 2.0 - 0.5
        center_x, center_y = joint
        threshold = 4.6052
        for g_y in range(self.heatmap_height):
            for g_x in range(self.heatmap_width):
                x = start + g_x * self.stride
                y = start + g_y * self.stride
                d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
                exponent = d2 / 2.0 / self.sigma / self.sigma
                if exponent > threshold:
                    continue

                heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
                if heatmap[g_y, g_x, plane_idx] > 1.0:
                    heatmap[g_y, g_x, plane_idx] = 1.0

    
    def get_image(self, filename, target_size=(96, 96)):        
        # load and preprocess the image datasets as well
        image = tf.keras.preprocessing.image.load_img(filename) 
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
        image_norm = image_arr/255.
        return image_norm
    
    def getKeyPointData(self, csvpath):
        # import pandas as pd
        df = pd.read_csv(csvpath)

        heatmaps = []
        images = []
        target_size = (self.input_img_width, self.input_img_height)

        for i, row in df.iterrows():
            filepath = row['Image']
            filename = filepath.split("\\")[-1]
            kps = [[row['p1X'], row['p1Y']],
                   [row['p2X'], row['p2Y']],
                   [row['p3X'], row['p3Y']],
                   [row['p4X'], row['p4Y']],
                   [row['p5X'], row['p5Y']],
                   [row['p6X'], row['p6Y']],
                   [row['p7X'], row['p7Y']],
                   [row['p8X'], row['p8Y']],
                   [row['p9X'], row['p9Y']]]
            image = self.get_image(filepath, target_size)
            heatmap = self.create_heatmap(kps)

            images.append(image)
            heatmaps.append(heatmap)

        return heatmaps, images
            


class Training:
    def __init__(self) -> None:
        pass

    def split_data(self, images, heatmaps):
        if len(images) != len(heatmaps):
            assert(len(images) == len(heatmaps))

        trainper = int(len(images) * 0.8)
        train_images = images[:trainper]
        train_heatmap = heatmaps[:trainper]

        test_images = images[trainper:]
        test_heatmap = heatmaps[trainper:]

        print(f"Shape of training images: {train_images.shape}")
        print(f"Shape of train labels: {train_heatmap.shape}")
        print(f"Shape of test images: {test_images.shape}")
        print(f"Shape of test labels: {test_heatmap.shape}")

        return train_images, train_heatmap, test_images, test_heatmap
    
    def decoder_block(self, x, squeeze):
        m1 = Conv2D(2*squeeze, (1,1), activation='relu')(x)
        m = Conv2D(squeeze, (3,3), activation='relu', padding='same')(m1)
        m = Conv2D(2*squeeze, (1,1), activation='relu')(m)
        return Add()([m1, m])
    
    def build_model(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape = (96, 96, 3),
                                                        alpha = 1.0,
                                                        include_top = False,
                                                        weights = "imagenet")

        # x = Conv2D(620, (1,1), activation='relu')(base_model.output) #decoder_block(base_model.output, 320)
        # x = UpSampling2D(size=(2, 2))(x)
        # x = Conv2D(240, (1,1), activation='relu')(x)#decoder_block(x, 80)
        # x = UpSampling2D(size=(2, 2))(x)
        # x = Conv2D(120, (1,1), activation='relu')(x) #decoder_block(x, 20)
        # x = UpSampling2D(size=(2, 2))(x)
        # x = Conv2D(11, (1,1))(x)

        x = self.decoder_block(base_model.output, 320)
        x = Dropout(0.2)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = self.decoder_block(x, 80)
        x = UpSampling2D(size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = self.decoder_block(x, 20)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(9, (1,1))(x)

        model = Model(inputs=[base_model.input], outputs=x)
        model.summary()

        return model 
    
    def heatmap_weighting_loss(self, y_true, y_pred):
        """Calculates the heatmap weighting loss.

        Args:
        y_true: A tensor of shape (batch_size, height, width, num_keypoints).
        y_pred: A tensor of shape (batch_size, height, width, num_keypoints).

        Returns:
        A tensor of shape (batch_size, 1).
        """

        # Calculate the weights for each pixel on the heatmap.
        weights = tf.reduce_sum(y_true, axis=3, keepdims=True)

        # Calculate the weighted squared error.
        weighted_squared_error = weights * tf.square(y_pred - y_true)

        # Return the average weighted squared error.
        return tf.reduce_mean(weighted_squared_error, axis=(1, 2))
    
    def train(self, train_images, train_heatmap, epochs=100, batch_size=10):
        checkpoint_path = "cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                         save_weights_only=True, 
                                                         save_best_only = True, 
                                                         verbose=1)
        model = self.build_model()
        model.compile(optimizer = 'adam', loss=self.heatmap_weighting_loss)
        model.fit(train_images, train_heatmap, epochs = epochs, batch_size = batch_size, validation_split = 0.2, callbacks = [cp_callback])
        model.save('shoesHM_V1.h5')
        print("model saved")



class Testing:

    def __init__(self) -> None:
        pass

    def get_keypoint_from_heatmap(self, heatmap):
        # Find the maximum value in the heatmap.
        max_value = np.max(heatmap)
        print("max_value : ", max_value)
        # Find the coordinates of the maximum value.
        (y, x) = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (x, y)


    def testmodel(self, modelpath, testimages):
        
        from tensorflow.keras.models import load_model
        import glob
        from keras.utils.generic_utils import get_custom_objects
        get_custom_objects().update({'heatmap_weighting_loss': Training().heatmap_weighting_loss})

        model = load_model(modelpath)
        print("model loaded Successfully", model)
        ip_shape = 96
        images = glob.glob(os.path.join(testimages, "*"))
        for im in images:
            filename = im.split("\\")[-1]
            image = cv2.imread(im)
            rimage = cv2.resize(image, (ip_shape, ip_shape))
            new_img = rimage/255.0
            pred = model.predict(new_img[np.newaxis, ...])

            
            res = np.reshape(pred, (9, 24, 24))
            print(res.shape)

            kp = []
            for i in range(9):
                print(res[i].shape)
                kp.append(self.get_keypoint_from_heatmap(res[i]))

            ratio_h = 96/24
            ratio_w = 96/24
            thickness = -1
            for k in kp:
                cv2.circle(rimage, (int(k[0]*ratio_w), int(k[1]*ratio_h)), 3, [255, 0, 0], thickness)
            cv2.imwrite("result1.jpg", rimage)
            
            # save image
            # cv2.imwrite(os.path.join(r"Training_set2\inferenceresults", filename), rimage)
            cv2.imshow("test", rimage)
            cv2.waitKey(100)