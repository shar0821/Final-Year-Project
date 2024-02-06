from flask import Flask, request, render_template, redirect, send_file, jsonify, g
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.data import Dataset, AUTOTUNE
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import numpy as np
import matplotlib.image

import base64
from jinja2 import Environment, FileSystemLoader
import glob
import os
import time
import pdfkit
from datetime import datetime
from io import BytesIO
from PIL import Image
import io
from flask_pymongo import PyMongo

app = Flask(__name__)
CORS(app)

MONGO_URI="mongodb+srv://sushaanth:stealthX33m@cluster0.iyhqlfn.mongodb.net/glaucomapredictionsystem?retryWrites=true&w=majority"
# MONGO_URI="mongodb://sushaanth:stealthX33m@ac-soirl34-shard-00-00.iyhqlfn.mongodb.net:27017,ac-soirl34-shard-00-01.iyhqlfn.mongodb.net:27017,ac-soirl34-shard-00-02.iyhqlfn.mongodb.net:27017/glaucomapredictionsystem?ssl=true&replicaSet=atlas-9748nb-shard-0&authSource=admin&retryWrites=true&w=majority"

mongodb_client = PyMongo(app, uri=MONGO_URI)
db = mongodb_client.db
# db = g._database = PyMongo(app).db

path_wkhtmltopdf = "./executables/wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
pdf_filename = ""
server_pdf_url = "https://localhost:5000/pdf"

# ------------------------------------------------- segmentation vars ------------------------------------------------- #

SAVE_PATH = "./validation/fundus/og"
PATH_TO_TEST = './validation/fundus/og/*' # path for segmentation
TEST_PATH="./validation/fundus/segmented/" # path for pretrained model

# TEST_IMAGES = np.array(glob.glob(PATH_TO_TEST + '/*'))

# exit(0)

# TEST_NORMAL_IMAGES = np.array(glob(os.path.join(PATH_TO_TEST, 'Images/normal/*')))
SEED = 2
EPSILON = 10 ** -4
STATELESS_RNG= tf.random.Generator.from_seed(SEED, alg='philox')
IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 4

# SHUFFLE_BUFFER = len(TRAIN_NORMAL_IMAGES) + len(TRAIN_GLAUCOMA_IMAGES)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ------------------------------------------------- custom objects needed for model ------------------------------------------------- #

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def dice(y, y_pred):
    epsilon = EPSILON
    numerator = 2 * tf.reduce_sum(y * y_pred, axis = [1, 2])
    denominator = tf.reduce_sum(y + y_pred, axis = [1, 2])
    dice = tf.reduce_mean((numerator + epsilon)/(denominator + epsilon))
    return 1 - dice

# ------------------------------------------------- loading models ------------------------------------------------- #

model = tf.keras.models.load_model('./models/siamesemodel_vgg_custom_200_SGD_1e-2_m0.9_binary.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
seg_model = tf.keras.models.load_model('./models/unet_segmentation_model.h5', custom_objects={"dice":dice})


# ------------------------------------------------- FSL ------------------------------------------------- #


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_png(byte_img)    
    # img = tf.image.resize(img, (128,64))
    img = img / 255
    img = img[:,:,0]
    return img

def predict_FSL():
    glaucoma_images = []
    normal_images = []
    test_imagex100 = []

    i = 0
    for file in os.listdir("./validation/oct/glaucoma"):
        glaucoma_images.append(preprocess(f'./validation/oct/glaucoma/{file}'))
        i += 1
        if i >= 100:
            break
    # print(len(glaucoma_images))

    i = 0
    for file in os.listdir("./validation/oct/normal"):
        normal_images.append(preprocess(f'./validation/oct/normal/{file}'))
        i += 1
        if i >= 100:
            break
    # print(len(normal_images))
    
    glaucoma_images = np.array(glaucoma_images)
    normal_images = np.array(normal_images)

    test_imagex100 = np.array([preprocess('./oct_img.png') for i in range(100)])

    y_hat_glaucoma = model.predict([test_imagex100, glaucoma_images])
    y_hat_normal = model.predict([test_imagex100, normal_images])

    return [y_hat_glaucoma, y_hat_normal]


# ------------------------------------------------- segmentation ------------------------------------------------- #


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
    test_images = np.array(glob.glob(PATH_TO_TEST))
    print(test_images)

    # test_ds_dice = Dataset.from_tensor_slices(test_images)\
    # .map(lambda x: load_image_with_masks(x, dice = True, test = True), num_parallel_calls = AUTOTUNE)\
    # .batch(batch_size = BATCH_SIZE)

    to_predict = Dataset.from_tensor_slices(test_images)\
        .map(lambda x: load_image(x), num_parallel_calls = AUTOTUNE)\
        .batch(batch_size = BATCH_SIZE)

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

    # print("middle_avg", middle_avg)

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

    # print("edge_avg", edge_avg)

    binarizer = (middle_avg + edge_avg) / 2

    return binarizer

def adaptive_binarizer(res, og_img, binarizer_threshold):
    og_img = np.array(list(og_img.as_numpy_iterator()))
    iters = 0
    ratio = 2
    while(ratio > 0.65 or ratio < 0.5):
        print("inside while")
        # print(type(res))
        # print(res.shape)
        # cv2.imwrite('nonbinarizedmask.png', res*255)   

        Binary = np.where(res > binarizer_threshold, 255, 0)

        num_zeros = np.count_nonzero(Binary==0)
        num_total = Binary.shape[0] * Binary.shape[1]

        ratio = num_zeros / num_total
        # print("ratio", ratio)

        if(ratio < 0.65 and ratio > 0.5):
            # cv2.imwrite('binarizedmask.png', Binary*255)

            # img = np.stack(list(og_img[i]))
            img = og_img[0]
            # print(img.shape)
            # exit(0)
            # img = img[0][0]
            final = Binary * img
            final = final / 255

            filename = "segmented_fundus.jpg"
            print("filename", filename)
            # matplotlib.image.imsave(f'./validation/fundus/segmented/{filename}', final)

            # print("final binarizer threshold", binarizer_threshold)
            return binarizer_threshold
            # exit(0)
            break

        if(ratio > 0.65):
            binarizer_threshold -= 0.015
        else:
            binarizer_threshold += 0.015

        iters += 1
        if(iters > 5):
            return binarizer_threshold

# ------------------------------------------------- pre trained model ------------------------------------------------- #

def classifyFundusImage():
    img = matplotlib.image.imread("./validation/fundus/segmented/segmented_fundus.png")
    img = img[:,:,:3]
    img = np.expand_dims(img, axis=0)
    # print("img shape", img.shape)

    img_width, img_height = 224, 224
    batch_size = 8 

    json_file = open('./models/VGG19_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('./models/VGG19_weights.h5')


    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    # test_generator = test_datagen.flow_from_directory(
    #     './validation/fundus/segmented',
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    vgg_preds = model.predict(img)

    print(vgg_preds)

    # vgg_model = load_model('./models/pre-trained-model.h5')
    # vgg_preds = vgg_model.predict(img)

    return vgg_preds

# ------------------------------------------------- report generation ------------------------------------------------- #

def generate_pdf(details):

    global pdf_filename

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report.html')

    date = datetime.now()

    html_template = template.render(
        patient_name=details["patient_name"],
        date=date,
        radiologist=details["radiologist"],
        oct_prob=details["oct_prob"],
        fundus_prob=details["fundus_prob"],
        weighted_avg_prob=details["weighted_avg_prob"],
        oct_path=details["oct_path"],
        fundus_path=details["fundus_path"]
    )

    pdf_options = {
        'page-size': 'A4',
        'margin-top': '1.00in',
        'margin-right': '1.00in',
        'margin-bottom': '0.50in',
        'margin-left': '1.00in',
        "enable-local-file-access": True
    }
    filename = f"./pdf/{details['patient_name']}_{str(time.time())}.pdf"
    filename = filename.replace(' ', '_')
    pdf_file = pdfkit.from_string(html_template, filename, options=pdf_options, configuration=config)

    pdf_filename = filename

def convertToBase64(image_path):
    with open(image_path, "rb") as img_file:
        img_string = base64.b64encode(img_file.read()).decode('utf-8')

    return img_string

# ------------------------------------------------- routes ------------------------------------------------- #

@app.route('/')
def index():
    return "<h1>Invalid Request</h1>"


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        
        global pdf_filename

        patient_name = request.form.get("patientName")
        radiologist_name = request.form.get("radiologistName")

        # converting base64 to image
        img1_string = request.form.get("image1String")
        img2_string = request.form.get("image2String")
        
        img1_string = img1_string[22:]
        img2_string = img2_string[22:]

        img1Data = base64.b64decode(img1_string)
        img2Data = base64.b64decode(img2_string)
        
        # saving the images
        filename = 'oct_img.png'
        with open(filename, 'wb') as f:
            f.write(img1Data)
        f.close()

        filename = 'fundus_img.png'
        with open(os.path.join(SAVE_PATH, filename), 'wb') as f:
            f.write(img2Data)
        f.close()
        
        
        # fsl computation
        res = predict_FSL()

        avg_prob_glaucoma1 = sum(res[0]) / len(res[0])
        avg_prob_normal1 = sum(res[1]) / len(res[1])

        avg_prob_glaucoma1 = round(avg_prob_glaucoma1[0], 2)
        avg_prob_normal1 = round(avg_prob_normal1[0], 2)

        # print(avg_prob_glaucoma1)
        # print(avg_prob_normal1)

        # unet segmentation
        res, og_img = segmentFundus()
        res = res[0]

        init_binarizer = find_binarizer(res*255)

        binarizer_threshold = adaptive_binarizer(res, og_img, init_binarizer)

        # res = res[0]
        # binarizer_threshold = 0.16
        Binary = np.where(res > binarizer_threshold, 255, 0)

        og_img = np.stack(list(og_img))
        og_img = og_img[0][0]
        final = Binary * og_img
        final = final / 255

        matplotlib.image.imsave('./validation/fundus/segmented/segmented_fundus.png', final)
    
        # pre trained computation
        vgg_preds = classifyFundusImage()
        # print("vgg preds shape: ", vgg_preds.shape)
        # print("vgg preds: ", vgg_preds)

        # vgg_pred_classes = np.argmax(vgg_preds, axis=1)
        # print("vgg pred classes: ", vgg_pred_classes)

        print("fsl: ", avg_prob_glaucoma1)
        print("unet + pre trained: ", vgg_preds[0][0])

        # vgg_preds[0][0] = -1

        vgg_preds[0][0] = ("%.17f" % vgg_preds[0][0]).rstrip('0').rstrip('.')

        oct_weight = 60
        fundus_weight = 40

        final_glaucoma_prob = ((fundus_weight * vgg_preds[0][0]) + (oct_weight * avg_prob_glaucoma1)) / 100
        final_glaucoma_prob *= 100
        
        # prediction_glaucoma = [1 if prediction > 0.5 else 0 for prediction in res[0]]
        # prediction_normal = [1 if prediction > 0.5 else 0 for prediction in res[1]]

        # print(prediction_glaucoma)
        # print(prediction_normal)

        # generating report
        details = {
            "patient_name": patient_name,
            "radiologist": radiologist_name,
            "oct_prob": round(avg_prob_glaucoma1*100),
            "fundus_prob": round(vgg_preds[0][0]*100),
            "weighted_avg_prob": round(final_glaucoma_prob),
            "oct_path": convertToBase64("./oct_img.png"),
            "fundus_path": convertToBase64("./validation/fundus/og/fundus_img.png"),
        }
        
        try:
            generate_pdf(details)
        except:
            pass

        # DB insertion
        # db.patientrecords.insert_one({
        #     "patient_name": patient_name,
        #     "radiologist": radiologist_name,
        #     "oct_prob": round(avg_prob_glaucoma1*100),
        #     "fundus_prob": round(vgg_preds[0][0]*100),
        #     "weighted_avg_prob": round(final_glaucoma_prob),
        #     "oct_path": convertToBase64("./oct_img.png"),
        #     "fundus_path": convertToBase64("./validation/fundus/og/fundus_img.png"),
        # })


        # deleting created files
        if os.path.exists("oct_img.png"):
            os.remove("oct_img.png")

        if os.path.exists("./validation/fundus/segmented/segmented_fundus.png"):
            os.remove("./validation/fundus/segmented/segmented_fundus.png")

        if os.path.exists("./validation/fundus/og/fundus_img.png"):
            os.remove("./validation/fundus/og/fundus_img.png")

        return jsonify(status = 200, glaucoma_prob=str(final_glaucoma_prob), pdf_url=server_pdf_url)
        # return jsonify(status = 200, glaucoma_prob1=str(avg_prob_glaucoma1), normal_prob1=str(avg_prob_normal1))
        
    return "<h1>Invalid Request</h1>"





@app.route('/classifyOCT', methods=['GET', 'POST'])
def classifyOCT():
    if request.method == 'POST':
        
        global pdf_filename

        patient_name = request.form.get("patientName")
        radiologist_name = request.form.get("radiologistName")

        # converting base64 to image
        img1_string = request.form.get("image1String")
        
        img1_string = img1_string[22:]

        img1Data = base64.b64decode(img1_string)
        
        # saving the images
        filename = 'oct_img.png'
        with open(filename, 'wb') as f:
            f.write(img1Data)
        f.close()
        
        
        # fsl computation
        res = predict_FSL()

        avg_prob_glaucoma1 = sum(res[0]) / len(res[0])
        avg_prob_normal1 = sum(res[1]) / len(res[1])

        avg_prob_glaucoma1 = round(avg_prob_glaucoma1[0], 2)
        avg_prob_normal1 = round(avg_prob_normal1[0], 2)

        # print(avg_prob_glaucoma1)
        # print(avg_prob_normal1)

        # generating report
        details = {
            "patient_name": patient_name,
            "radiologist": radiologist_name,
            "oct_prob": round(avg_prob_glaucoma1*100),
            "fundus_prob": "-",
            "weighted_avg_prob": "-",
            "oct_path": convertToBase64("./oct_img.png"),
            "fundus_path": "",
        }
        
        try:
            generate_pdf(details)
        except:
            pass

        # DB insertion
        db.patientrecords.insert_one({
            "patient_name": patient_name,
            "radiologist": radiologist_name,
            "oct_prob": round(avg_prob_glaucoma1*100),
            "fundus_prob": -1,
            "weighted_avg_prob": -1,
            "oct_base64": convertToBase64("./oct_img.png"),
            "fundus_base64": "",
        })


        # deleting created files
        if os.path.exists("oct_img.png"):
            os.remove("oct_img.png")

        return jsonify(status = 200, glaucoma_prob=str(round(avg_prob_glaucoma1*100)))
        # return jsonify(status = 200, glaucoma_prob1=str(avg_prob_glaucoma1), normal_prob1=str(avg_prob_normal1))
        
    return "<h1>Invalid Request</h1>"


@app.route('/classifyFundus', methods=['GET', 'POST'])
def classifyFundus():
    if request.method == 'POST':
        global pdf_filename

        patient_name = request.form.get("patientName")
        radiologist_name = request.form.get("radiologistName")

        # converting base64 to image
        img1_string = request.form.get("image1String")
        
        img1_string = img1_string[22:]

        img1Data = base64.b64decode(img1_string)
        

        filename = 'fundus_img.png'
        with open(os.path.join(SAVE_PATH, filename), 'wb') as f:
            f.write(img1Data)
        f.close()


        # unet segmentation
        res, og_img = segmentFundus()

        res = res[0]

        binarizer_threshold = 0.16
        Binary = np.where(res > binarizer_threshold, 255, 0)

        og_img = np.stack(list(og_img))
        og_img = og_img[0][0]
        final = Binary * og_img
        final = final / 255

        matplotlib.image.imsave('./validation/fundus/segmented/segmented_fundus.png', final)
    
        # pre trained computation
        vgg_preds = classifyFundusImage()
        # print("vgg preds shape: ", vgg_preds.shape)
        # print("vgg preds: ", vgg_preds)

        # vgg_pred_classes = np.argmax(vgg_preds, axis=1)
        # print("vgg pred classes: ", vgg_pred_classes)

        print("unet + pre trained: ", vgg_preds[0][0])
        
        # prediction_glaucoma = [1 if prediction > 0.5 else 0 for prediction in res[0]]
        # prediction_normal = [1 if prediction > 0.5 else 0 for prediction in res[1]]

        # print(prediction_glaucoma)
        # print(prediction_normal)

        # generating report
        details = {
            "patient_name": patient_name,
            "radiologist": radiologist_name,
            "oct_prob": "-",
            "fundus_prob": round(vgg_preds[0][0]*100),
            "weighted_avg_prob": "-",
            "oct_path": "",
            "fundus_path": convertToBase64("./validation/fundus/og/fundus_img.png"),
        }
        
        try:
            generate_pdf(details)
        except:
            pass

        # DB insertion
        db.patientrecords.insert_one({
            "patient_name": patient_name,
            "radiologist": radiologist_name,
            "oct_prob": "-",
            "fundus_prob": round(vgg_preds[0][0]*100),
            "weighted_avg_prob": "-",
            "oct_path": "",
            "fundus_path": convertToBase64("./validation/fundus/og/fundus_img.png"),
        })


        # deleting created files
        if os.path.exists("oct_img.png"):
            os.remove("oct_img.png")

        if os.path.exists("./validation/fundus/segmented/segmented_fundus.png"):
            os.remove("./validation/fundus/segmented/segmented_fundus.png")

        if os.path.exists("./validation/fundus/og/fundus_img.png"):
            os.remove("./validation/fundus/og/fundus_img.png")

        return jsonify(status = 200, glaucoma_prob=str(vgg_preds[0][0]*100), pdf_url=server_pdf_url)
        # return jsonify(status = 200, glaucoma_prob1=str(avg_prob_glaucoma1), normal_prob1=str(avg_prob_normal1))
        
    return "<h1>Invalid Request</h1>"


@app.route('/pdf', methods=['GET'])
def pdf():
    print("pdf filename: ", pdf_filename)
    return send_file(pdf_filename, attachment_filename='report.pdf')