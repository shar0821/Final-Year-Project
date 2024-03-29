import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import backend as k
import numpy as np



modelName = 'VGG19'
img_width, img_height = 224, 224
batch_size = 8  


# default paths
model_name = 'model.json'
model_weights = modelName + '_weights.h5'


def classify(trained_model_dir, test_data_dir, results_dir):

    json_file = open(os.path.join(trained_model_dir, model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(trained_model_dir, model_weights))

    test_datagen = ImageDataGenerator(rescale=1. / 255
                                      )

    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      shuffle=False)

    y_probabilities = model.predict_generator(test_generator,
                                              steps = np.math.ceil(test_generator.samples / float(test_generator.batch_size))
                                              )

    filenames = test_generator.filenames
    print(len(filenames))
    print(y_probabilities)
    write_submission(y_probabilities, results_dir, filenames)


def write_submission(preds, results_dir, filenames):

    sub_fn = results_dir + 'results_' + modelName

    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image_name, Glaucoma, Not Glaucoma\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.5f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")


if __name__ == '__main__':

    test_dir = 'images/'
    model_dir = 'models/' + modelName + '/'
    results_dir = 'results/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    classify(model_dir, test_dir, results_dir)
    k.clear_session()
