import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import model_from_json
import numpy as np

def load_hist_model_w (model_name):

    # load json and create model
    json_file_l = open(model_name + '_train.json', 'r')
    loaded_model_json = json_file_l.read()
    json_file_l.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_name+ '_train_w.h5')
    print("Loaded model from disk")

    return loaded_model

#Basic configuration settings.
img_width, img_height = 224, 224
batch_size = 16
test_path = '/Users/dk_macpro/Documents/cars/val/'

#Using the model VGG16_SGD_Unlocked_NoReg_NoInit, LR 0.0003
model_name = "VGG16_SGD_UnLocked_NoReg_NoInit2_lr"
learn_rate = 0.0003

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(test_path,
                                                   target_size=(img_height, img_width),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

filenames = test_generator.filenames
num_samples = len(filenames)

test_generator.reset()

#Loading Pretrained model
loaded_model= load_hist_model_w(model_name)
#Choose SGD, compile the loaded model wit hSGD
sgd = SGD(lr=learn_rate, decay=1e-4, momentum=0.9, nesterov=True)
loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#Run Evaluate Generator to get the score and present score in %
score = loaded_model.evaluate_generator(test_generator,verbose=1, steps=num_samples)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
