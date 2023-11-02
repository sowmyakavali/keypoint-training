import cv2 
import numpy as np 
import pandas as pd 
from math import sin, cos, pi
import matplotlib.pyplot as plt


class config:
    horizontal_flip = False
    rotation_augmentation = False
    brightness_augmentation = True
    shift_augmentation = True
    random_noise_augmentation = False

    rotation_angles = [45, 90]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)
    pixel_shifts = [30]    # Horizontal & vertical shift amount in pixels (includes shift from all 4 corners)

    NUM_EPOCHS = 1000
    BATCH_SIZE = 32
    input_size = 98


def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([config.input_size-coor if idx%2==0 else coor for idx, coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints


def rotate_augmentation(images, keypoints):
    rotated_images = []
    rotated_keypoints = []
    print("Augmenting for angles (in degrees): ")
    for angle in config.rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((config.input_size//2, config.input_size//2), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (config.input_size, config.input_size), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - config.input_size//2   # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint), 2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += config.input_size//2  # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
            
    return np.reshape(rotated_images,(-1, config.input_size, config.input_size, 1)), rotated_keypoints


def alter_brightness(images, keypoints):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.5, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.8, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))


# def shift_images(images, keypoints):
#     shifted_images = []
#     shifted_keypoints = []
#     for shift in config.pixel_shifts:    # Augmenting over several pixel shift values
#         for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
#             M = np.float32([[1,0,shift_x],[0,1,shift_y]])
#             for image, keypoint in zip(images, keypoints):
#                 shifted_image = cv2.warpAffine(image, M, (config.input_size, config.input_size), flags=cv2.INTER_CUBIC)
#                 shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
#                 if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<config.input_size.0):
#                     shifted_images.append(shifted_image.reshape(config.input_size, config.input_size,1))
#                     shifted_keypoints.append(shifted_keypoint)
#     shifted_keypoints = np.clip(shifted_keypoints, 0.0, config.input_size.0)
#     return shifted_images, shifted_keypoints


def add_noise(images):
    noisy_images = []
    for image in images:
        noisy_image = cv2.add(image, 0.008*np.random.randn(config.input_size, config.input_size, 1))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]
        noisy_images.append(noisy_image.reshape(config.input_size, config.input_size, 1))
    return noisy_images


def plot_sample(image, keypoint, axis, title):
    image = image.reshape(config.input_size, config.input_size)
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)


def load_images(image_data):
    images = []
    for idx, sample in image_data.iterrows():
        image = np.array(sample['Image'].split(' '), dtype=int)
        # print(image.shape)
        image = np.reshape(image, (config.input_size, config.input_size))
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


def preprocess(train_data, imagesize):

    # train_data = pd.read_csv(train_file)

    clean_train_data = train_data.dropna()
    print("clean_train_data shape:", np.shape(clean_train_data))
    clean_train_images = load_images(clean_train_data)
    print("Shape of clean_train_images:", np.shape(clean_train_images))
    clean_train_keypoints = load_keypoints(clean_train_data)
    print("Shape of clean_train_keypoints:", np.shape(clean_train_keypoints))

    # test_images = load_images(test_data)
    # print("Shape of test_images:", np.shape(test_images))

    train_images = clean_train_images
    train_keypoints = clean_train_keypoints
    # fig, axis = plt.subplots()
    # plot_sample(clean_train_images[19], clean_train_keypoints[19], axis, "Sample image & keypoints")

    # unclean_train_data = train_data.fillna(method = 'ffill')
    # print("unclean_train_data shape:", np.shape(unclean_train_data))
    # unclean_train_images = load_images(unclean_train_data)
    # print("Shape of unclean_train_images:", np.shape(unclean_train_images))
    # unclean_train_keypoints = load_keypoints(unclean_train_data)
    # print("Shape of unclean_train_keypoints:", np.shape(unclean_train_keypoints))

    # train_images = np.concatenate((train_images, unclean_train_images))
    # train_keypoints = np.concatenate((train_keypoints, unclean_train_keypoints))

    if config.horizontal_flip:
        flipped_train_images, flipped_train_keypoints = left_right_flip(clean_train_images, clean_train_keypoints)
        print("Shape of flipped_train_images:", np.shape(flipped_train_images))
        print("Shape of flipped_train_keypoints:", np.shape(flipped_train_keypoints))
        train_images = np.concatenate((train_images, flipped_train_images))
        train_keypoints = np.concatenate((train_keypoints, flipped_train_keypoints))
        fig, axis = plt.subplots()
        plot_sample(flipped_train_images[19], flipped_train_keypoints[19], axis, "Horizontally Flipped") 

    if config.rotation_augmentation:
        rotated_train_images, rotated_train_keypoints = rotate_augmentation(clean_train_images, clean_train_keypoints)
        print("Shape of rotated_train_images:", np.shape(rotated_train_images))
        print("Shape of rotated_train_keypoints:", np.shape(rotated_train_keypoints))
        train_images = np.concatenate((train_images, rotated_train_images))
        train_keypoints = np.concatenate((train_keypoints, rotated_train_keypoints))
        fig, axis = plt.subplots()
        plot_sample(rotated_train_images[11], rotated_train_keypoints[11], axis, "Rotation Augmentation")
    
    if config.brightness_augmentation:
        altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(clean_train_images, clean_train_keypoints)
        print("Shape of altered_brightness_train_images:", np.shape(altered_brightness_train_images))
        print("Shape of altered_brightness_train_keypoints:", np.shape(altered_brightness_train_keypoints))
        train_images = np.concatenate((train_images, altered_brightness_train_images))
        train_keypoints = np.concatenate((train_keypoints, altered_brightness_train_keypoints))
        fig, axis = plt.subplots()
        plot_sample(altered_brightness_train_images[1], altered_brightness_train_keypoints[1], axis, "Increased Brightness") 
        fig, axis = plt.subplots()
        plot_sample(altered_brightness_train_images[len(altered_brightness_train_images)//2+19], altered_brightness_train_keypoints[len(altered_brightness_train_images)//2+19], axis, "Decreased Brightness") 

    # if config.shift_augmentation:
    #     shifted_train_images, shifted_train_keypoints = shift_images(clean_train_images, clean_train_keypoints)
    #     print("Shape of shifted_train_images:", np.shape(shifted_train_images))
    #     print("Shape of shifted_train_keypoints:", np.shape(shifted_train_keypoints))
    #     train_images = np.concatenate((train_images, shifted_train_images))
    #     train_keypoints = np.concatenate((train_keypoints, shifted_train_keypoints))
    #     fig, axis = plt.subplots()
    #     plot_sample(shifted_train_images[19], shifted_train_keypoints[19], axis, "Shift Augmentation")

    if config.random_noise_augmentation:
        noisy_train_images = add_noise(clean_train_images)
        print("Shape of noisy_train_images:", np.shape(noisy_train_images))
        train_images = np.concatenate((train_images, noisy_train_images))
        train_keypoints = np.concatenate((train_keypoints, clean_train_keypoints))
        fig, axis = plt.subplots()
        plot_sample(noisy_train_images[19], clean_train_keypoints[19], axis, "Random Noise Augmentation")

    return train_images, train_keypoints
