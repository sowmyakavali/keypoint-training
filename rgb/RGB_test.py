import os
import cv2 
import glob
import numpy as np 
from tensorflow.keras.models import load_model

modelpath = r"D:\FinalDataset_224x224\ew_16b_224s_6kps.h5"
model = load_model(modelpath)
print("model loaded Successfully", model)

size = 224

inputsource = r"FinalDataset_96x96\RGB_Images"
if os.path.isdir(inputsource):
    images = glob.glob(os.path.join(inputsource, "*"))

    for im in images: 
        filename = im.split("\\")[-1]
        image = cv2.imread(im)
        rimage = cv2.resize(image, (size, size))

        new_img = rimage/255.0
        new_img2 = np.reshape(new_img, (-1, size, size, 3))

        pred = model(new_img2)
        xs, ys = pred[0][0::2], pred[0][1::2]

        for X, Y in zip(xs, ys):
            rimage = cv2.circle(rimage, (int(X), int(Y)), 1, (0, 0, 255), 2)
        # save image
        cv2.imwrite(os.path.join(r"Training_set2\inferenceresults", filename), rimage)
        cv2.imshow("test", rimage)
        cv2.waitKey(100)

elif os.path.isfile(inputsource) or type(inputsource) == int:
    if os.path.isfile(inputsource) == True:
        savepath = inputsource.split("\\")[-1].split(".")[0] +'_results.mp4'
        vid_cap =  cv2.VideoCapture(inputsource)
    else:
        savepath = 'results.mp4'
        vid_cap =  cv2.VideoCapture(inputsource, cv2.CAP_DSHOW)
    w, h, fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), vid_cap.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, img = vid_cap.read()

        rimage = cv2.resize(img, (size, size))
        new_img2 = np.reshape(rimage, (-1, size, size, 1))

        pred = model(new_img2)
        xs, ys = pred[0][0::2], pred[0][1::2]

        kpind = 0
        for X, Y in zip(xs, ys):
            rimage = cv2.circle(rimage, (int(X), int(Y)), 1, (0, 0, 255), 2)
            rimage = cv2.putText(rimage, str(kpind), (int(X)-5, int(Y)-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            kpind += 1

        rimage = cv2.cvtColor(rimage, cv2.COLOR_RGB2BGR)
        vid_writer.write(rimage)
        # save image
        cv2.imshow("test", rimage)
        cv2.waitKey(10)
    vid_writer.release()
