"""
    read csv from visualization script
    1. crop bboxes.
    2. resize bboxes to 96x96.
    3. normalize keypoints to 96x96 image.
    4. generate final csv for keypoint training.
"""
import os 
import cv2
import tqdm
import pandas as pd

def process(keypoints, xmin, ymin, cropped_image, resize, fileName, imheight, imwidth, imtype):
    # reset keypoints to cropped image
    cropped_img_kps = []
    for i in range(0, len(keypoints), 2):       
        if imtype == 'fullimage':
            kpx, kpy = keypoints[i], keypoints[i+1]
        else:
            kpx, kpy = keypoints[i] - xmin, keypoints[i+1] - ymin
        cropped_img_kps.append((kpx, kpy))

    # adjust keypoints to 96x96 image
    shapes = cropped_image.shape
    Height, Width = shapes[0], shapes[1]
    # print("Height, Width", Height, Width)

    resized_img_kps = []
    normalized_keypoints = []
    for (x, y) in cropped_img_kps:
        try:
            if imtype == 'fullimage':
                xkp, ykp = int((x * resize) / imwidth), int((y * resize) / imheight)
            else:
                xkp, ykp = int((x * resize) / Width), int((y * resize) / Height)
            resized_img_kps.append(xkp)
            resized_img_kps.append(ykp)

            normalized_keypoints.append(xkp)
            normalized_keypoints.append(ykp)
        except ZeroDivisionError:
            pass
            # print(fileName, Height, Width)

    # crop object and save
    rgb_resized_image = cv2.resize(cropped_image, (resize, resize))
    gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray_resized_image = cv2.resize(gray_img, (resize, resize))
    imlist = []
    for i in gray_resized_image.tolist():
        imlist = imlist + i
    imlist = " ".join(str(val) for val in imlist)  

    resized_img_kps.append(imlist)

    grayvis_image = gray_resized_image.copy()
    rgbvis_image = rgb_resized_image.copy()
    for i in range(0, len(normalized_keypoints), 2):
        grayvis_image = cv2.circle(grayvis_image, (int(normalized_keypoints[i]), int(normalized_keypoints[i+1])), 2, (255,0,0), 2)
        rgbvis_image = cv2.circle(rgbvis_image, (int(normalized_keypoints[i]), int(normalized_keypoints[i+1])), 2, (255,0,0), 2)

    return resized_img_kps, gray_resized_image, grayvis_image, normalized_keypoints, rgb_resized_image, rgbvis_image


def main(csv_path, 
         input_images_path, 
         padding,
         verifycount, 
         resize = 96, 
         hmresize = 224,
         mainfolder = r"Dataset",
         noOfpoints = 3):
    """
        1. Load csv file generated with columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', '2dkeypoints']
        2. Load image, bbox, 2d keypoints
        3. visualize all annotations
        4. generate a csv file which u need for training.
        input : images path, csv file
        output : final csv file, cropped images
    """
    
    df = pd.read_csv(csv_path)

    # RGB
    rgbData = os.path.join(mainfolder, "RGB_Images")
    rgbVerify = os.path.join(mainfolder, "RGB_Images_Verify")
    rgbcsvfile = os.path.join(mainfolder, "RGBFinalData.csv")

    # Gray
    grayData = os.path.join(mainfolder, "Gray_Images")
    grayVerify = os.path.join(mainfolder, "Gray_Images_Verify")
    graycsvfile = os.path.join(mainfolder, "grayFinalData.csv")

    # Gray
    heatmapData = os.path.join(mainfolder, "Hm_Images")
    heatmapVerify = os.path.join(mainfolder, "Hm_Images_Verify")
    heatmapcsvfile = os.path.join(mainfolder, "HmFinalData.csv")
    
    # create directories
    directories = [grayData, grayVerify, rgbData, rgbVerify, heatmapData, heatmapVerify]
    for i in directories:
        if not os.path.isdir(i):
            os.makedirs(i)

    # declare lists
    rgbcsv = []
    graycsv = []
    heatmapcsv = []
    imtype = 'cropped'
     
    # column names
    columns = []
    for i in range(1, noOfpoints+1):
         pointX = 'p' + str(i) + 'X'
         pointY = 'p' + str(i) + 'Y'
         columns.append(pointX)
         columns.append(pointY)
         
    columns.append('Image')
    fn = ''
    for ind, row in tqdm.tqdm(df.iterrows()):

        if row['filename'] != fn:
            index = '0'
        else:
            index = '1'
        
        if not os.path.isfile(row['filename']):
             filepathh = os.path.join(input_images_path, row['filename'] )
             fileName = row['filename']
             print(fileName)
        else:
            filepathh = row['filename'] 
            fileName = row['filename'].split("\\")[-1]

        image = cv2.imread(filepathh)
        imheight, imwidth, _ = image.shape
        [xmin, ymin, xmax, ymax] = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        cropped_image = image[ymin-padding:ymax+padding, xmin-padding:xmax+padding]
        keypoints = row['keypoints']
        keypointsvalues = [float(i) for i in keypoints[1:-1].split(", ")]

        shapes = cropped_image.shape
        Height, Width = shapes[0], shapes[1]
        if (Width<=0 or Height<=0) or len(keypointsvalues) < noOfpoints*2:
             continue
        grayList, grayImg, grayVis, rgbList, rgbImg, rgbVis = process(keypointsvalues, xmin, ymin, cropped_image, resize, fileName, imheight, imwidth, imtype)
        
        rgb_save_filename = "rgb_" + fileName.split(".")[0] + "_" + index + ".jpg"
        gray_save_filename = "gray_" + fileName.split(".")[0] + "_" + index + ".jpg"
        rgbsavepath = os.path.join(rgbData, rgb_save_filename)
        graysavepath = os.path.join(grayData, gray_save_filename)
        
        cv2.imwrite( rgbsavepath, rgbImg)
        cv2.imwrite( graysavepath, grayImg)
        
        rgbList.append(rgbsavepath)
        rgbcsv.append(rgbList)
        graycsv.append(grayList)

        # Heatmap task 
        O_height, O_width, _ = image.shape
        hm_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hm_resize = cv2.resize(hm_gray, (hmresize, hmresize))
        hm_imlist = []
        for i in hm_resize.tolist():
            hm_imlist = hm_imlist + i
        hm_imlist = " ".join(str(val) for val in hm_imlist)

        hm_img_kps = []
        im = hm_resize.copy()
        for i in range(0, len(keypointsvalues), 2):               
            kpx, kpy = keypointsvalues[i], keypointsvalues[i+1]
            try:
                xkp, ykp = int((kpx * hmresize) / O_width), int((kpy * hmresize) / O_height)
                hm_img_kps.append(xkp)
                hm_img_kps.append(ykp)
                im = cv2.circle(im, (xkp, ykp ), 2, (255,0,0), 2)
            except:
                continue
        if len(hm_img_kps) == 2*noOfkpoints:
            hm_img_kps.append(hm_imlist)
            heatmapcsv.append(hm_img_kps)
        else:
            continue

        hm_save_filename = "hm_" + fileName.split(".")[0] + "_" + index + ".jpg"
        hmsavepath = os.path.join(heatmapData, hm_save_filename)
        cv2.imwrite( hmsavepath, hm_resize )
        
        if ind<verifycount:
            rgbsavepath_verify = os.path.join(rgbVerify, rgb_save_filename)
            graysavepath_verify = os.path.join(grayVerify, gray_save_filename)
            hmsavepath_verify = os.path.join(heatmapVerify, hm_save_filename)

            cv2.imwrite( rgbsavepath_verify, rgbVis)
            cv2.imwrite( graysavepath_verify, grayVis)
            cv2.imwrite( hmsavepath_verify, im )

        fn = row['filename']

    train_rgb = pd.DataFrame(rgbcsv, columns=columns)
    train_rgb.to_csv(rgbcsvfile, index=False)

    train_gray = pd.DataFrame(graycsv, columns=columns)
    train_gray.to_csv(graycsvfile, index=False)

    train_heatmap = pd.DataFrame(heatmapcsv, columns=columns)
    train_heatmap.to_csv(heatmapcsvfile, index=False)
                     

if __name__ == "__main__":
    mainpath = r'Datasets1to8'
    csv_path = os.path.join(mainpath, 'finalData.csv')
    images_path = os.path.join(mainpath, "images") # If absolute path is already there not required otherwise u need to mention images path
    noOfkpoints = 9
    padding = 0
    resize = 96 #regression model input shape
    hmresize = 224 # Heatmap image size
    verifycount = 100
    mainpath = os.path.join(mainpath, f"FinalDataset_{resize}x{resize}")

    main(csv_path, images_path, padding, verifycount, resize = resize, hmresize = hmresize, mainfolder = mainpath, noOfpoints = noOfkpoints)