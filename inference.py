import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, UpSampling2D
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, AveragePooling2D, Activation
from classification_models.keras import Classifiers
# import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def resnet_generator(input_shape = (64, 64, 3)):
    ResNet34, _ = Classifiers.get('resnet34')
    # base_model = ResNet34(input_shape = input_shape, weights = 'imagenet', include_top = False)
    base_model = ResNet34(input_shape = input_shape, include_top = False)
    del ResNet34

    inp = base_model.input
    conv4 = base_model.layers[129].output

    convm = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv4)
    convm = LeakyReLU(0.2)(convm)
    convm = Conv2D(1024, 3, activation = 'relu', padding = 'same')(convm)
    convm = BatchNormalization()(convm)
    convm = LeakyReLU(0.2)(convm)
    convm = AveragePooling2D()(convm)
    convm = Dropout(0.4)(convm)

    up4 = Conv2DTranspose(512, kernel_size = (3, 3), strides = (2, 2), activation= 'relu', padding = 'same')(convm)
    conc4 = concatenate([conv4, up4])
    conv6 = Conv2D(512, 3, padding = 'same')(conc4)
    conv6 = LeakyReLU(0.2)(conv6)
    conv6 = Conv2D(512, 3, padding = 'same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(0.2)(conv6)
    #conv6 = Activation('relu')(conv6)

    up3 = Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), activation= 'relu', padding = 'same')(conv6)
    conv3 = base_model.layers[74].output
    conc3 = concatenate([conv3, up3])
    conv7 = Conv2D(256, 3, padding = 'same')(conc3)
    conv7 = LeakyReLU(0.2)(conv7)
    conv7 = Conv2D(256, 3, padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(0.2)(conv7)
    #conv7 = Activation('relu')(conv7)

    up2 = Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), activation= 'relu', padding = 'same')(conv7)
    conv2 = base_model.layers[37].output
    conc2 = concatenate([conv2, up2])
    conv8 = Conv2D(128, 3, padding = 'same')(conc2)
    #conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(0.2)(conv8)
    conv8 = Conv2D(128, 3, padding = 'same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(0.2)(conv8)
    #conv8 = Activation('relu')(conv8)

    up1 = Conv2DTranspose(64, kernel_size = (3, 3), strides = (2, 2), activation= 'relu', padding = 'same')(conv8)
    conv1 = base_model.layers[5].output
    conc1 = concatenate([conv1, up1])
    conv9 = Conv2D(64, 3, padding = 'same')(conc1)
    conv9 = LeakyReLU(0.2)(conv9)
    conv9 = Conv2D(64, 3, padding = 'same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(0.2)(conv9)

    up0 = Conv2DTranspose(32, kernel_size = (3, 3), strides=(2, 2), activation = 'relu', padding="same")(conv9)
    conv10 = Conv2D(32, 3, padding = 'same')(up0)
    conv10 = LeakyReLU(0.2)(conv10)
    conv10 = Conv2D(32, 3, padding = 'same')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv10 = LeakyReLU(0.2)(conv10)

    conv10 = Conv2D(2, 1, activation = 'tanh', padding = 'same')(conv10)
        
    model = Model(inputs = inp, outputs=  conv10)
    return model

def read_grayscale(gray_batch, shape):
    batch_images = []
    for img in gray_batch:
        #print(img)
        img = cv2.resize(img, (shape[0], shape[1]))
        img = (img - 127.5) / 127.5
        #print(img.shape)
        img = np.resize(img, (shape[0], shape[1], 1))
        batch_images.append(img)
    return np.array(batch_images, dtype = np.float32)

def get_color_from_lab(gray_batch, ab_batch, shape):
    batch_images = []
    for i in range(len(gray_batch)):
        img = np.zeros(shape)
        #img[:, :, 0] = gray_batch[i]
        #img[:, :, 1:] = ab_batch[i]
        img = ab_batch[i]
        img = img.astype('uint8')
        #img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = cv2.resize(img, (shape[0], shape[1]))
        img = (img - 127.5) / 127.5
        batch_images.append(img)
    return np.array(batch_images, dtype = np.float32)

def read_input_images(image_names, shape):
    batch_images = []
    for name in image_names:
        img = cv2.imread(os.path.join(img_path, name))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (shape[0], shape[1]))
        img = (img - 127.5) / 127.5
        img = np.resize(img, (shape[0], shape[1], 1))
        batch_images.append(img)
    return np.array(batch_images, dtype = np.float32)
    
def get_target_images(image_names, shape):
    batch_targets = []
    for name in image_names:
        img = cv2.imread(os.path.join(img_path, name))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (shape[0], shape[1]))
        img = (img  - 127.5) / 127.5
        batch_targets.append(img)
    return np.array(batch_targets, dtype = np.float32)

def ab2rgb(gray_img, ab_img, shape):
    batch_images = []
    for i in range(len(gray_img)):
        img = np.zeros(shape)
        img[:, :, 0] = (gray_img[i]).astype('uint8')
        img[:, :, 1:] = (ab_img[i]).astype('uint8')
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        batch_images.append(img)
    return np.array(batch_images, dtype = np.uint8)

def color(images, req_shape):
    # input_img = read_grayscale(images, input_size)
    input_img = images
    input_img = np.expand_dims(input_img, 0)
    input_img = np.expand_dims(input_img, 3)
    inp_img = np.repeat(input_img, 3, -1)
    inp_img = (inp_img - 127.5) / 127.5
    print(inp_img.shape, input_img.shape)
    predict = g.predict(inp_img)[0]
    predict = np.ceil(((predict * 127.5) + 127.5)).astype('uint8')
    input_img = np.squeeze(input_img)
    # input_img = np.ceil(((input_img * 127.5) + 127.5)).astype('uint8')
    # print(input_img.max())
    # print(input_img.shape, predict.shape)
    # print(inp_img.max(), predict.max())
    colorized = ab2rgb([input_img], [predict], req_shape)[0]
    print(colorized.max())
    predict = predict / 255.0
    # predict = cv2.cvtColor(predict, cv2.COLOR_RGB2LAB)
    #print(predict.shape)
#     plt.imshow(predict)
    return colorized


print("Image Colorization.")
test_img_path = input("Enter the path for the image : ")

t_img = cv2.imread(test_img_path)
t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
original_t_shape = t_img.shape
print("Original Image Shape :", original_t_shape)

if original_t_shape[0] > 1600:
    new_height = 1600
else:
    new_height = round(original_t_shape[0] / 32) * 32
if original_t_shape[1] > 1600:
    new_width = 1600
else:
    new_width = round(original_t_shape[1] / 32) * 32
new_t_shape = (new_height, new_width, 3)
print("Resized Image Shape :", new_t_shape)

t_img = cv2.resize(t_img, (new_t_shape[1], new_t_shape[0]))
g = resnet_generator(new_t_shape)
g.load_weights('./weights/final_generator.h5')

fixed_image = color(t_img, new_t_shape)

file_name = input("Enter the file name for the colorized image : ")
# saving the images
from PIL import Image
o_img = Image.fromarray(fixed_image)
print(np.repeat(np.expand_dims(t_img, 2), 3, -1).shape)
c_img = Image.fromarray(np.hstack(((np.repeat(np.expand_dims(t_img, 2), 3, -1)), o_img)))
o_img.save('./images_outputs/' + file_name + '_output.jpg')
c_img.save('./images_outputs/' + file_name + '_combined.jpg')
