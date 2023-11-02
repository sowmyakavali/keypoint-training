import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

import keras, sys, time, os, warnings, cv2

from keras.models import *
from keras.layers import *

import numpy as np
import pandas as pd
from tensorflow.keras import layers, callbacks, applications

from skimage import transform
from skimage.transform import SimilarityTransform, AffineTransform
import random 

from sklearn.utils import shuffle


def gaussian_k(x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hm(height, width, landmarks, s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1,-1]):
             
                hm[:,:,i] = gaussian_k(landmarks[i][0],
                                        landmarks[i][1],
                                        s,height, width)
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm
    
def get_y_as_heatmap(df,height,width, sigma):
    
    columns_lmxy = df.columns[:-1] ## the last column contains Image
    columns_lm = [] 
    for c in columns_lmxy:
        c = c[:-1]
        if c not in columns_lm:
            columns_lm.extend([c])
    
    print("columns_lm : ", columns_lm)
    y_train = []
    for i in range(df.shape[0]):
        landmarks = []
        for colnm in columns_lm:
            x = df[colnm + "X"].iloc[i]
            y = df[colnm + "Y"].iloc[i]
            if np.isnan(x) or np.isnan(y):
                x, y = -1, -1
            landmarks.append([x,y])
            
        y_train.append(generate_hm(height, width, landmarks, sigma))
    y_train = np.array(y_train)
    
    
    return(y_train,df[columns_lmxy],columns_lmxy)

def load(fname, test=False, width=96, height=96, sigma=5):
    """
    load test/train data
    cols : a list containing landmark label names.
           If this is specified, only the subset of the landmark labels are 
           extracted. for example, cols could be:
           
          [left_eye_center_x, left_eye_center_y]
            
    return: 
    X:  2-d numpy array (Nsample, Ncol*Nrow)
    y:  2-d numpy array (Nsample, Nlandmarks*2) 
        In total there are 15 landmarks. 
        As x and y coordinates are recorded, u.shape = (Nsample,30)
    y0: panda dataframe containins the landmarks
       
    """
    
    
    df = pd.read_csv(fname)[:10]

    
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))


    myprint = df.count()
    myprint = myprint.reset_index()
    print(myprint)  
    ## row with at least one NA columns are removed!
    ## df = df.dropna()  
    df = df.fillna(-1)

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    if not test:  # labels only exists for the training data
        y, y0, nm_landmark = get_y_as_heatmap(df, height, width, sigma)
        X, y, y0 = shuffle(X, y, y0, random_state=42)  # shuffle data   
        y = y.astype(np.float32)
    else:
        y, y0, nm_landmark = None, None, None
    
    return X, y, y0, nm_landmark

def load2d(filename, test=False, width=96, height=96, sigma=5):

    re = load(filename, test, width, height, sigma)
    X  = re[0].reshape(-1, width, height, 1)
    y, y0, nm_landmarks = re[1:]
    
    return X, y, y0, nm_landmarks


def transform_img(data,
                  loc_w_batch=2,
                  max_rotation=0.01,
                  max_shift=2,
                  max_shear=0,
                  max_scale=0.01,mode="edge"):
    '''
    data : list of numpy arrays containing a single image
    e.g., data = [X, y, w] or data = [X, y]
    X.shape = (height, width, NfeatX)
    y.shape = (height, width, Nfeaty)
    w.shape = (height, width, Nfeatw)
    NfeatX, Nfeaty and Nfeatw can be different
    
    affine transformation for a single image
    
    loc_w_batch : the location of the weights in the fourth dimention
    [,,,loc_w_batch]
    '''
    scale = (np.random.uniform(1-max_scale, 1 + max_scale),
             np.random.uniform(1-max_scale, 1 + max_scale))
    rotation_tmp = np.random.uniform(-1*max_rotation, max_rotation)
    translation = (np.random.uniform(-1*max_shift, max_shift),
                   np.random.uniform(-1*max_shift, max_shift))
    shear = np.random.uniform(-1*max_shear, max_shear)
    tform = AffineTransform(
            scale=scale,#,
            ## Convert angles from degrees to radians.
            rotation=np.deg2rad(rotation_tmp),
            translation=translation,
            shear=np.deg2rad(shear)
        )
    
    for idata, d in enumerate(data):
        if idata != loc_w_batch:
            ## We do NOT need to do affine transformation for weights
            ## as weights are fixed for each (image,landmark) combination
            data[idata] = transform.warp(d, tform,mode=mode)
    return data

def transform_imgs(data, lm, 
                   loc_y_batch = 1, 
                   loc_w_batch = 2):
    '''
    data : list of numpy arrays containing a single image
    e.g., data = [X, y, w] or data = [X, y]
    X.shape = (height, width, NfeatX)
    y.shape = (height, width, Nfeaty)
    w.shape = (height, width, Nfeatw)
    NfeatX, Nfeaty and Nfeatw can be different
    
    affine transformation for a single image
    '''
    Nrow  = data[0].shape[0]
    Ndata = len(data) 
    data_transform = [[] for i in range(Ndata)]
    for irow in range(Nrow):
        data_row = []
        for idata in range(Ndata):
            data_row.append(data[idata][irow])
        ## affine transformation
        data_row_transform = transform_img(data_row,
                                          loc_w_batch)
        ## horizontal flip
        data_row_transform = horizontal_flip(data_row_transform,
                                             lm,
                                             loc_y_batch,
                                             loc_w_batch)
        
        for idata in range(Ndata):
            data_transform[idata].append(data_row_transform[idata])
    
    for idata in range(Ndata):
        data_transform[idata] = np.array(data_transform[idata])
    
    
    return(data_transform)

def horizontal_flip(data,lm,loc_y_batch=1,loc_w_batch=2):  
    '''
    flip the image with 50% chance
    
    lm is a dictionary containing "orig" and "new" key
    This must indicate the potitions of heatmaps that need to be flipped  
    landmark_order = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
                      "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}
                      
    data = [X, y, w]
    w is optional and if it is in the code, the position needs to be specified
    with loc_w_batch
    
    X.shape (height,width,n_channel)
    y.shape (height,width,n_landmarks)
    w.shape (height,width,n_landmarks)
    '''
    lo, ln = np.array(lm["orig"]), np.array(lm["new"])

    assert len(lo) == len(ln)
    if np.random.choice([0,1]) == 1:
        return(data)
    
    for i, d in enumerate(data):
        d = d[:, ::-1,:] 
        data[i] = d


    data[loc_y_batch] = swap_index_for_horizontal_flip(
        data[loc_y_batch], lo, ln)

    # when horizontal flip happens to image, we need to heatmap (y) and weights y and w
    # do this if loc_w_batch is within data length
    if loc_w_batch < len(data):
        data[loc_w_batch] = swap_index_for_horizontal_flip(
            data[loc_w_batch], lo, ln)
    return(data)

def swap_index_for_horizontal_flip(y_batch, lo, ln):
    '''
    lm = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
          "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}
    lo, ln = np.array(lm["orig"]), np.array(lm["new"])                  
    '''
    y_orig = y_batch[:,:, lo]
    y_batch[:,:, lo] = y_batch[:,:, ln] 
    y_batch[:,:, ln] = y_orig
    return(y_batch)

def get_model():
    input_height, input_width = 96, 96
    ## output shape is the same as input
    output_height, output_width = input_height, input_width 
    n = 32*5
    nClasses = 3
    nfmp_block1 = 64
    nfmp_block2 = 128

    IMAGE_ORDERING =  "channels_last" 
    img_input = Input(shape=(input_height, input_width, 1)) 

    # Encoder Block 1
    x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    block1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
        
    # Encoder Block 2
    x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(block1)
    x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
        
    ## bottoleneck    
    o = (Conv2D(n, (int(input_height/4), int(input_width/4)), activation='relu' , padding='same', name="bottleneck_1", data_format=IMAGE_ORDERING))(x)
    o = (Conv2D(n , ( 1 , 1 ) , activation='relu' , padding='same', name="bottleneck_2", data_format=IMAGE_ORDERING))(o)


    ## upsamping to bring the feature map size to be the same as the one from block1
    ## o_block1 = Conv2DTranspose(nfmp_block1, kernel_size=(2,2),  strides=(2,2), use_bias=False, name='upsample_1', data_format=IMAGE_ORDERING )(o)
    ## o = Add()([o_block1, block1])
    ## output = Conv2DTranspose(nClasses, kernel_size=(2,2),  strides=(2,2), use_bias=False, name='upsample_2', data_format=IMAGE_ORDERING )(o)

    ## Decoder Block
    ## upsampling to bring the feature map size to be the same as the input image i.e., heatmap size
    output = Conv2DTranspose(nClasses, kernel_size=(4, 4),  strides=(4, 4), use_bias=False, name='upsample_2', data_format = IMAGE_ORDERING )(o)

    ## Reshaping is necessary to use sample_weight_mode="temporal" which assumes 3 dimensional output shape
    ## See below for the discussion of weights
    output = Reshape((output_width*input_height*nClasses, 1))(output)
    model = Model(img_input, output)
    model.summary()

    return model


def find_weight(y_tra):
    '''
    :::input:::
    
    y_tra : np.array of shape (N_image, height, width, N_landmark)
    
    :::output::: 
    
    weights : 
        np.array of shape (N_image, height, width, N_landmark)
        weights[i_image, :, :, i_landmark] = 1 
                        if the (x,y) coordinate of the landmark for this image is recorded.
        else  weights[i_image, :, :, i_landmark] = 0

    '''
    weight = np.zeros_like(y_tra)
    count0, count1 = 0, 0
    for irow in range(y_tra.shape[0]):
        for ifeat in range(y_tra.shape[-1]):
            if np.all(y_tra[irow,:,:,ifeat] == 0):
                value = 0
                count0 += 1
            else:
                value = 1
                count1 += 1
            weight[irow,:,:,ifeat] = value
    print("N landmarks={:5.0f}, N missing landmarks={:5.0f}, weight.shape={}".format(
        count0,count1,weight.shape))
    return(weight)


def flatten_except_1dim(weight,ndim=2):
    '''
    change the dimension from:
    (a,b,c,d,..) to (a, b*c*d*..) if ndim = 2
    (a,b,c,d,..) to (a, b*c*d*..,1) if ndim = 3
    '''
    n = weight.shape[0]
    if ndim == 2:
        shape = (n,-1)
    elif ndim == 3:
        shape = (n,-1,1)
    else:
        print("Not implemented!")
    weight = weight.reshape(*shape)
    return(weight)

from matplotlib import pyplot as plt

if __name__ == "__main__":
    FTRAIN = r"D:\Hand\Heatmap\fullImage_3kps_train.csv"
    FTEST  =  r"D:\Hand\Heatmap\fullImage_3kps_testkps.csv"

    sigma = 5
    X_train, y_train, y_train0, nm_landmarks = load2d(FTRAIN, test=False, sigma=sigma)
    X_test,  y_test, _, _ = load2d(FTEST, test=False, sigma=sigma)
    print(y_train)
    print( X_train.shape, y_train.shape, y_train0.shape)
    print( X_test.shape, y_test)

    Nplot = y_train.shape[3] + 1
    # for i in range(2):
    #     fig = plt.figure(figsize=(20, 6))
    #     ax = fig.add_subplot(2, int(Nplot/2), 1)
    #     ax.imshow(X_train[i,:,:,0], cmap="gray")
    #     ax.set_title("input")
    #     for j, lab in enumerate(nm_landmarks[::2]):
    #         ax = fig.add_subplot(2,Nplot/2, j+2)
    #         ax.imshow(y_train[i,:,:,j], cmap="gray")
    #         ax.set_title(str(j) +"\n" + lab[:-2] )
    #     plt.show()

    landmark_order = {"orig" : [0, 1, 2], "new"  : [0, 2, 1]}
    
    prop_train = 0.9
    Ntrain = int(X_train.shape[0]*prop_train)
    X_tra, y_tra, X_val, y_val = X_train[:Ntrain], y_train[:Ntrain], X_train[Ntrain:], y_train[Ntrain:]
    
    w_tra = find_weight(y_tra)

    weight_val = find_weight(y_val)
    weight_val = flatten_except_1dim(weight_val)
    y_val_fla  = flatten_except_1dim(y_val, ndim=3) 

    ## print("weight_tra.shape={}".format(weight_tra.shape))
    print("weight_val.shape={}".format(weight_val.shape))
    print("y_val_fla.shape={}".format(y_val_fla.shape))

    x_batch, y_batch, w_batch = transform_imgs([X_tra, y_tra, w_tra], landmark_order)
    # If you want no data augementation, comment out the line above and uncomment the comment below:
    # x_batch, y_batch, w_batch = X_tra,y_tra, w_batch 
    w_batch_fla = flatten_except_1dim(w_batch, ndim=2)
    y_batch_fla = flatten_except_1dim(y_batch, ndim=3)

    model = get_model()

    es = callbacks.EarlyStopping(monitor='val_loss', 
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
    # model.compile(optimizer='adam', 
    #                     loss='mse', 
    #                     metrics=['acc'])
    
    checkpoint_path = r"D:\Hand\Heatmap\fullImage_wristintermediate_19042023.h5"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_freq='epoch',
                                                        period=5,
                                                        verbose=1)
    
    model.compile(loss='mse', optimizer="rmsprop", sample_weight_mode="temporal")
    nb_epochs = 300
    batch_size = 64
    const = 1
    hist = model.fit(x_batch,
                     y_batch_fla*const,
                     sample_weight = w_batch_fla,
                     validation_data=(X_val, y_val_fla*const, weight_val),
                     batch_size=batch_size,
                     epochs=nb_epochs,
                     verbose=1,
                     callbacks=[es, rlp, cp_callback]
                     )
    
    # score = model.evaluate( X_val, y_val)[1]
    # print("Final Score  : ", score)

    model.save(r"D:\Hand\Heatmap\fullImage_wistkp_model_19042023.h5")