import numpy as np
import glob
from tensorflow.data import Dataset, AUTOTUNE
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image
import cv2
import os
import json

BATCH_SIZE = 4
IMAGE_SIZE=(224,224)
PATH_TO_TEST = './Test/*'
EPSILON = 10**-4

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def dice(y, y_pred):
    epsilon = EPSILON
    numerator = 2 * tf.reduce_sum(y * y_pred, axis = [1, 2])
    denominator = tf.reduce_sum(y + y_pred, axis = [1, 2])
    dice = tf.reduce_mean((numerator + epsilon)/(denominator + epsilon))
    return 1 - dice

seg_model = tf.keras.models.load_model('Model1.h5', custom_objects={"dice":dice})

def load_image(path):
    
    # filename = tf.strings.split(path, sep = '.')[-2]
    # filename = tf.strings.split(filename, sep = '/')[-1]
    # filename = tf.strings.split(filename, sep = '\\')[-1] # added

    # print(path)

    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels = 3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    
    return image

def segmentFundus():
    global test_images
    test_images = np.array(glob.glob(PATH_TO_TEST))
    # print(test_images)

    # test_ds_dice = Dataset.from_tensor_slices(test_images)\
    # .map(lambda x: load_image_with_masks(x, dice = True, test = True), num_parallel_calls = AUTOTUNE)\
    # .batch(batch_size = BATCH_SIZE)

    to_predict = Dataset.from_tensor_slices(test_images)\
        .map(lambda x: load_image(x), num_parallel_calls = AUTOTUNE)\
        .batch(batch_size = 1000)

    seg_predict = seg_model.predict(to_predict)
    return seg_predict, to_predict
   
def find_binarizer(img):
    # img is np array
    # img = img/255

    middle_pixels = []
    for i in range(0, 100, 10):
        middle_pixels.append(img[(img.shape[0]//2)+(i//5)][img.shape[0]//2]+(i//5))
        middle_pixels.append(img[(img.shape[0]//2)-(i//5)][img.shape[0]//2]-(i//5))
        middle_pixels.append(img[(img.shape[0]//2)+(i//5)][img.shape[0]//2]-(i//5))
        middle_pixels.append(img[(img.shape[0]//2)-(i//5)][img.shape[0]//2]+(i//5))
        middle_pixels.append(img[(img.shape[0]//2)][img.shape[0]//2]+i)
        middle_pixels.append(img[(img.shape[0]//2)][img.shape[0]//2]-i)
        middle_pixels.append(img[(img.shape[0]//2)+i][img.shape[0]//2])
        middle_pixels.append(img[(img.shape[0]//2)-i][img.shape[0]//2])

    middle_avg = 0

    for i in range(len(middle_pixels)):
        middle_avg += middle_pixels[i]
    
    middle_avg = middle_avg // len(middle_pixels)

    middle_avg = np.array(middle_avg)
    middle_avg = middle_avg / 255

    print("middle_avg", middle_avg)

    # -----------------------------------------------------------------

    edge_pixels = []
    for i in range(0, 100, 10):
        edge_pixels.append(img[0][i])
        edge_pixels.append(img[i][0])
        edge_pixels.append(img[img.shape[0]-1][-i])
        edge_pixels.append(img[-i][img.shape[0]-1])

    edge_avg = 0

    for i in range(len(edge_pixels)):
        edge_avg += edge_pixels[i]
    
    edge_avg = edge_avg // len(edge_pixels)

    edge_avg = np.array(edge_avg)
    edge_avg = edge_avg / 255

    print("edge_avg", edge_avg)

    binarizer = (middle_avg + edge_avg) / 2

    return binarizer



res, og_img = segmentFundus()
og_img = np.array(list(og_img.as_numpy_iterator()))


i = 0
for i in range(len(res)):
    binarizer_threshold = find_binarizer(res[i]*255)
    # binarizer_threshold -= 0.12
    # binarizer_threshold = 0.06

    ratio = 2
    while(ratio > 0.65 or ratio < 0.5):
        print("inside while")
        # print(type(res))
        # print(res.shape)
        # cv2.imwrite('nonbinarizedmask.png', res*255)   

        Binary = np.where(res[i] > binarizer_threshold, 255, 0)

        num_zeros = np.count_nonzero(Binary==0)
        num_total = Binary.shape[0] * Binary.shape[1]

        ratio = num_zeros / num_total
        # print("ratio", ratio)

        if(ratio < 0.65 and ratio > 0.5):
            # cv2.imwrite('binarizedmask.png', Binary*255)

            # img = np.stack(list(og_img[i]))
            img = og_img[0][i]
            # print(img.shape)
            # exit(0)
            # img = img[0][0]
            final = Binary * img
            final = final / 255

            filename = test_images[i].split("\\")[-1]
            print("filename", filename)
            matplotlib.image.imsave(f'./acrima_segmented/normal/{filename}', final)

            print("final binarizer threshold", binarizer_threshold)
            # exit(0)
            break

        if(ratio > 0.65):
            binarizer_threshold -= 0.015
        else:
            binarizer_threshold += 0.015

        # print("binarizer trial", binarizer_threshold)
        # print("number of zeros: ", num_zeros)
        # print("number of zeros / total: ", num_zeros/num_total)