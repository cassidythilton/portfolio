import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import mlflow
import mlflow.keras
import tensorflow as tf
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, concatenate
from tensorflow.keras import Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import os
import math
import matplotlib.pyplot as plt
import cv2

plt.style.use("dark_background")


def readData(dirname='src'):
    cnt=1
    for dirname, _, filenames in os.walk(dirname):
        for filename in filenames:
            if(cnt==2):
                break;        
            fileLocation=os.path.join(dirname, filename)
            print(fileLocation)
            cnt+=1

    df=pd.read_csv(fileLocation)   
    return df

def plotSampleImages():
    image_directory = 'img'
    all_image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]

    sample_image_paths = all_image_paths[:100]
    #sample_image_paths = all_image_paths

    num_images = len(sample_image_paths)
    grid_size = math.ceil(math.sqrt(num_images))
    rows, cols = grid_size, grid_size

    plt.figure(figsize=(20, 10))

    plt.subplots_adjust(wspace=0, hspace=0)

    for i, img_path in enumerate(sample_image_paths):
        img = cv2.imread(img_path)
        imgR = cv2.resize(img, (400, 300))

        plt.subplot(rows, cols, i+1)
        plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

def createANN(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    # return our model
    return model

def createCNN(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    #inpute shape: (64,64,3)
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Conv2D(16, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

def generatePredictionRange():
    f = np.random.randint(0, 15450)
    u = f + 17
    
    for i in range(f, u):
        plt.figure()  

        attr_sample = df.loc[df['image_id'] == i]

        image_sample = cv2.imread(f'img/{i}.jpg')
        sample_resized = cv2.resize(image_sample, (64, 64))

        plt.imshow(cv2.cvtColor(image_sample, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        X1_final = np.zeros(4, dtype='float32')
        X1_final[0] = float(attr_sample['n_citi'].iloc[0]) / citiM
        X1_final[1] = float(attr_sample['bed'].iloc[0]) / bM
        X1_final[2] = float(attr_sample['bath'].iloc[0]) / bathM
        X1_final[3] = float(attr_sample['sqft'].iloc[0]) / sqftM
        
        y_ground_truth = float(attr_sample['price'].iloc[0])

        X2_final = sample_resized / 255.0

        y_pred = model.predict([np.reshape(X1_final, (1, 4)), np.reshape(X2_final, (1, 64, 64, 3))])

        print(i)
        print(f"Actual Price: ${int(y_ground_truth):,}")
        print(f"Predicted Price:  ${int(y_pred * priceM):,}")
        plt.show()  

    def setAttributes(df):
    xHouseAttributes = df[['n_citi', 'bed', 'bath', 'sqft', 'price']].copy()

    bM=max(xHouseAttributes['bed'])
    sqftM=max(xHouseAttributes['sqft'])
    priceM=max(xHouseAttributes['price'])
    bathM=max(xHouseAttributes['bath'])
    citiM=max(xHouseAttributes['n_citi'])


    for column in xHouseAttributes.columns:
        max_value = xHouseAttributes[column].max()
        xHouseAttributes[column] = xHouseAttributes[column] / max_value
    
    return bM, sqftM, priceM, bathM, citiM, xHouseAttributes
