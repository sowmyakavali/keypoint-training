import tensorflow as tf
import os 
import glob 
import cv2 
import time
import numpy as np 

model_path=r'D:\FinalDataset_96x96\MobileNet0.63_default.tflite'


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

size = 96

inputsource = r"D:\ML-projects\tryon-training\eyewearData\ts2"
if os.path.isdir(inputsource):
    images = glob.glob(os.path.join(inputsource, "*"))

    for im in images:
        Annos = {}
        filename = im.split("\\")[-1]
        image = cv2.imread(im)
        print("image.shape", image.shape)
        rimage = cv2.resize(image, (size, size))

        new_img = (rimage/255.0).astype(np.float32)
        new_img2 = np.reshape(new_img, (-1, size, size, 3))

        # Give input to tflite model
        t1 = time.time()
        interpreter.set_tensor(input_details[0]['index'], new_img2)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]['index'])
        print(time.time() - t1)
        # print(pred)
        xs, ys = pred[0][0::2], pred[0][1::2]
        print("preds == ", im, xs, ys)

        kps = []
        for X, Y in zip(xs, ys):
            rimage = cv2.circle(rimage, (int(X), int(Y)), 1, (0, 0, 255), 2)
            kps.append((int(X), int(Y)))

        # save image
        cv2.imwrite(os.path.join(r"D:\testresults", filename), rimage)
        cv2.imshow("test", rimage)
        cv2.waitKey(10)
