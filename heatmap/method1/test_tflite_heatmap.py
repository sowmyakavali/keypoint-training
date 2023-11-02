import os
import cv2
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# for gpu growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def heatmap_to_coord(heatmap, image_width, image_height):
    """
    Convert a heatmap keypoint to x and y coordinates.
    
    Args:
        heatmap (numpy.ndarray): A 2D array representing the heatmap.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
    
    Returns:
        A tuple (x, y) representing the x and y coordinates of the keypoint.
    """
    max_value = np.max(heatmap)
    max_index = np.where(heatmap == max_value)
    y, x = max_index[0][0], max_index[1][0]
    x = x / heatmap.shape[1] * image_width
    y = y / heatmap.shape[0] * image_height
    return (x, y)

if __name__ == "__main__":
    # Access image 
    save = True
    show = True
    input = 0 
    ind = 0 

    # Load models
    t1 = time.time()
    keypoint_model = r"c:\Users\user\Downloads\shoes_heatmap.tflite"

    detection_Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(keypoint_model) 
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    savepath = 'results.mp4'
    vid_cap =  cv2.VideoCapture(0, cv2.CAP_DSHOW)
    w, h, fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), vid_cap.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    size = 224
    while True:
        ret, img = vid_cap.read()

        rimage = cv2.resize(img, (size, size))
        grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_img = (grayimage/255.0).astype(np.float32)
        # new_img2 = np.reshape(new_img, (-1, size, size, 1))
        print(input_details[0]['index'], '=========')
        interpreter.set_tensor(input_details[0]['index'], new_img)
        interpreter.invoke()
        res = interpreter.get_tensor(output_details[0]['index'])
        
        nClasses = 9
        y_pred = res.reshape(-1, size, size, nClasses)
        Nlandmark = y_pred.shape[-1]

        newimage = rimage.reshape((size, size, 1))

        coords = []
        for j in range(nClasses):
            hm = y_pred[0, :, :, j]
            xycoord = heatmap_to_coord(hm, size, size)
            Xcord, Ycord = int(xycoord[0]), int(xycoord[1])
            newimage = cv2.circle(newimage, (Xcord, Ycord), 5, (0, 0, 255), 5)
        # print(newimage)
        im = cv2.cvtColor(newimage, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite(os.path.join("results", f"image_{inde}.jpg"), im)

        cv2.imshow("test", im)
        cv2.waitKey(10)