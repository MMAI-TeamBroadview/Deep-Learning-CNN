import warnings
warnings.filterwarnings("ignore")
import os
import shutil as sh
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
from pickle import load

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.optimizers import SGD, RMSprop, Adam

from keras.applications.vgg16 import VGG16
from keras.models import model_from_json

import time
import datetime

# from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")

#Mac
meta_path = '/devkit/'
train_path = '/cars_train/'
test_path = '/cars_test/'
val_path = '/cars_val/'

# ===========================================================================================

def load_data(meta_path):
    '''
    Loads metadata and create dataframs for Cars, Train, and Test
    '''
    # LOAD Metadata
    print('---Loading Data--')
    cars_meta = sio.loadmat(meta_path + 'cars_meta.mat')
    cars_train_annos = sio.loadmat(meta_path + 'cars_train_annos.mat')
    cars_test_annos = sio.loadmat(meta_path + 'cars_test_annos_withlabels.mat')
    # print(cars_meta.keys())
    # print(cars_train_annos.keys())
    # print(cars_test_annos.keys())

    # Create dataframe with Car labels metadata
    data = [[index+1, name[0]] for index, name in enumerate (cars_meta['class_names'].ravel())]
    cars = pd.DataFrame(data, columns =['class','name'])

    # Create dataframe with Train  metadata
    data = [[x[0].ravel()[0], # bbox_x1
            x[1].ravel()[0], #bbox_x2
            x[2].ravel()[0], #bbox_y1
            x[3].ravel()[0], #bbox_y2
            x[4].ravel()[0], #class
            x[5].ravel()[0]] #fname
             for x in cars_train_annos['annotations'].ravel()]
    train = pd.DataFrame(data, columns = ['bbox_x1','bbox_x2','bbox_y1','bbox_y2','class','fname'])
    train = pd.merge(train, cars, on='class', how = 'left') #Merge name info from cars df

    # Create dataframe with test  metadata
    data = [[x[0].ravel()[0], # bbox_x1
            x[1].ravel()[0], #bbox_x2
            x[2].ravel()[0], #bbox_y1
            x[3].ravel()[0], #bbox_y2
            x[4].ravel()[0], #class
            x[5].ravel()[0]] #fname
                for x in cars_test_annos['annotations'].ravel()]
    test = pd.DataFrame(data, columns = ['bbox_x1','bbox_x2','bbox_y1','bbox_y2','class','fname'])
    test = pd.merge(test, cars, on='class', how = 'left') #Merge name info from cars df

    del(data, cars_meta,cars_test_annos,cars_train_annos)

    return cars, train, test

# Moves the images to Train and Test folder structure
def organize_data (df, path):
    for index, row in df.iterrows():
        os.makedirs(path + row['name'], exist_ok=True)
        sh.move(path + row['fname'], path + row['name'] + '/' + row['fname'])
    print('Move Complete!')


def split_test_data (test_path, val_path):
    '''
    This method splits the test data in to test and val sets (~50/50 split)
    '''
    class_counter = 0
    move_counter = 0
    skip_counter = 0
    for folder in os.listdir(test_path):        # Each Class
        if not folder.startswith('.'):          # Ignore hidden folders in mac (eg. .DS_store)
            fpath = test_path + folder + "/"    # Test_directory
            counter = 0
            for file in os.listdir(fpath):      # Each image per class
                if not file.startswith('.'):
                    counter += 1
                    if counter % 2 == 0:        # Move every other file (~50%)
                        os.makedirs(val_path + folder + '/', exist_ok=True)
                        sh.move(fpath + file, val_path + folder + '/' + file)
                        move_counter += 1
                    else:
                        skip_counter += 1
            class_counter += 1
    print(class_counter, ' classes processed')
    print(move_counter + skip_counter, ' files processed')
    print('>', move_counter, ' files moved to cars_val')
    print('>', skip_counter, ' files skipped')


def data_generator(train_path, test_path, batch_size, img_height, img_width):
    '''
    Generate images with transformations.
    For train apply rescaling, horizontal flip, shear, and rotation
    For test just apply rescaling
    '''

    print('---Creating Train and Test Generators---')
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       shear_range= 0.2,
                                       fill_mode='nearest',
                                       rotation_range=30)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Create Generators for Train and Test
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_path,
                                                       target_size=(img_height, img_width),
                                                       batch_size=batch_size,
                                                       class_mode='categorical')
    return train_generator, test_generator


# Standard CNN Architecture
def standardConvNet(dropout, img_height, img_width, optimizer):
    '''
    Standard CNN - 3 convolution layers, followed by 2 fully connected layers
    '''
    print('---Building Standard ConvNet Model---')
    model = Sequential()
    # Three layers of Convolution
    model.add(Conv2D(32, kernel_size=3,
                     activation='relu',
                     padding='same',
                     input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=2))
    model.add(Conv2D(64, kernel_size=3,
                     activation='relu',
                     padding='same',
                     input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=2))
    model.add(Conv2D(128, kernel_size=3,
                     activation='relu',
                     padding='same',
                     input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=2))
    # Two fully connected layers before output
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(196, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary

    return model

def VGG16model(dropout, img_height, img_width, optimizer, trainable = False):
    '''
    Transfer Learning with VGG16 - adding two fully connected layers after VGG16 for training
    '''
    print('---Build VGG16 Model---')

    model = Sequential()

    model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    # If trainable = True, then the last block (5) are trainable. All other layers are not trainable
    if trainable:
        for layer in model_vgg.layers[:-4]:
            layer.trainable = False
    # If trainable = False, then all layers are not trainable
    else:
        for layer in enumerate(model_vgg.layers):
            layer[1].trainable = False

    model.add(model_vgg)
    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(196, activation='softmax',
                    kernel_initializer='random_uniform',
                    bias_initializer='random_uniform',
                    name='predictions'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('-----VGG Layers-----')
    for layer in enumerate(model_vgg.layers):
        print(layer[1].trainable)
    print('-----Model Layers-----')
    for layer in enumerate(model.layers):
        print(layer[1].trainable)

    model_vgg.summary()
    model.summary()

    return model

##Save History and Model (incl Weights)
def save_hist_model_w(model, history, model_name):
    # Save History file
    with open('history_' + model_name + '_pickle', 'wb') as file_pi:
        pickle.dump(history, file_pi)

    # Save model to json
    model_json = model.to_json()
    with open(model_name + '_train.json', "w") as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights(model_name + '_train_w.h5')
    print("Saved model to disk")


def load_hist_model_w (model_name):
    #Load History file
    with open('history_' +model_name +'_pickle', 'rb') as handle: # loading old history
        savedHistory = load(handle)

    # load json and create model
    json_file_l = open(model_name + '_train.json', 'r')
    loaded_model_json = json_file_l.read()
    json_file_l.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_name+ '_train_w.h5')
    print("Loaded model from disk")

    return savedHistory, loaded_model

def plotHistory(history2, model_name):
    print('a')
    loss_list = [s for s in history2.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history2.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history2.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history2.history.keys() if 'acc' in s and 'val' in s]
    print('b')
    epochs_plt = range(1, len(history2.history[loss_list[0]]) + 1)
    print('c')
    plt.clf()
    plt.figure(1)
    print('d')
    for l in loss_list:
        plt.plot(epochs_plt, history2.history[l], 'b',
                 label='Training loss (' + str(str(format(history2.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs_plt, history2.history[l], 'r',
                 label='Validation loss (' + str(str(format(history2.history[l][-1], '.5f')) + ')'))
    print('e')
    plt.title('Loss ' +  model_name + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('Loss_' + model_name + '.png')  # Save Loss graph
    plt.figure(2)
    print('f')
    for l in acc_list:
        plt.plot(epochs_plt, history2.history[l], 'b',
                 label='Training accuracy (' + str(format(history2.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs_plt, history2.history[l], 'r',
                 label='Validation accuracy (' + str(format(history2.history[l][-1], '.5f')) + ')')
    print('g')
    plt.title('Accuracy ' + model_name + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    print('h')
    plt.legend()
    plt.show()
    plt.savefig('Acc_' + model_name + '.png')  # Save Accuacy Graph

# ===========================================================================================

# cars, train, test = load_data(meta_path)
# train.to_csv('./train.csv', index = False)
# test.to_csv('./test.csv', index = False)
# cars.to_csv('./cars.csv', index = False)

# organize_data(train, train_path)
# organize_data(test, test_path)
# split_test_data(test_path, val_path)


img_width, img_height = 224, 224
batch_size = 16
epochs = 100
earlystop_patience = 10
sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
rms = RMSprop(decay=1e-4, lr=0.001)
adam = Adam()
dropout = 0.5
# regularizer = regularizers.l2(0.01)

'''
Parameters to Change:

Dropout = 0.25, 0.5 (default)
learning rate = 0.0001, 0.001 (default) 

'''

train_generator, test_generator = data_generator(train_path, test_path, batch_size, img_height, img_width)

#model =  standardConvNet(dropout, img_height, img_width, sgd)
model = VGG16model(dropout,img_height, img_width, sgd, False)

earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=earlystop_patience)

ts = time.time()
train_start_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

history = model.fit_generator(
        train_generator,
        verbose= 1,
        steps_per_epoch= 8144 // batch_size,
        epochs=epochs,
        validation_data = test_generator,
        validation_steps= 4064 // batch_size,
        callbacks = [earlystop])

ts = time.time()
train_finish_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Start: ", train_start_time)
print("Epochs: ", epochs)
print("Batch size: ", batch_size)
print("early stopping patience:", earlystop_patience)
print("Finish:", train_finish_time)

save_hist_model_w (model, history, 'ConvNet')
plotHistory(history, 'ConvNet')
