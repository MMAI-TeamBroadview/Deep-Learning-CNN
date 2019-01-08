import warnings
warnings.filterwarnings("ignore")
import os
import shutil as sh
import pandas as pd
import numpy as np
# import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

#Mac
meta_path = '/devkit/'
train_path = '/cars_train/'
test_path = '/cars_test/'
val_path = '/cars_val/ß'

# ===========================================================================================

# Loads metadata and create dataframs for Cars, Train, and Test
def load_data(meta_path, train_path, test_path):
    # LOAD Metadata
    print('---Start: Loading Data--')
    cars_meta = sio.loadmat(meta_path + 'cars_meta.mat')
    cars_train_annos = sio.loadmat(meta_path + 'cars_train_annos.mat')
    cars_test_annos = sio.loadmat(meta_path + 'cars_test_annos_withlabels.mat')
    # print(cars_meta.keys())
    # print(cars_train_annos.keys())
    # print(cars_test_annos.keys())

    # Create dataframe with Car labels metadata
    data = [[index+1, name[0]] for index, name in enumerate (cars_meta['class_names'].ravel())]
    cars = pd.DataFrame(data, columns =['class','name'])
    del(data)

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
    del(data)

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

    print('---Done: Loading Data--')
    return cars, train, test


# Moves the images to Train and Test folder structure
def organize_data (df, path):
    for index, row in df.iterrows():
        os.makedirs(path + row['name'], exist_ok=True)
        sh.move(path + row['fname'], path + row['name'] + '/' + row['fname'])
    print('Move Complete!')

def split_test_data (test_path, val_path):
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

# Generate images with tranformations
def data_generator(train_path, test_path, batch_size, img_height, img_width):
    # For train, we will apply some transformation to generate new batches of data for each class.
    print('---Start: Creating Train and Test Generators---')
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True,
                                       # width_shift_range=0.3,
                                       # height_shift_range=0.3,
                                       fill_mode='nearest',
                                       rotation_range=30)
    # For test, just rescaling
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
    print('---Done: Creating Train and Test Generators---')
    return train_generator, test_generator


# Standard CNN Architecture
def standardConvNet(dropout, img_height, img_width):
    print('---Start: Creating Standard ConvNet Model---')
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
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(196, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print('---Done: Creating Standard ConvNet Model---')
    return model

# serializes the trained model and its weights
def serializeModel(model, fileName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(fileName + ".h5")
    print("Saved model to disk")

# ===========================================================================================

cars, train, test = load_data(meta_path, train_path, test_path)
# train.to_csv('./train.csv', index = False)
# test.to_csv('./test.csv', index = False)
# cars.to_csv('./cars.csv', index = False)

# organize_data(train, train_path)
# organize_data(test, test_path)
# split_test_data(test_path, val_path)

img_width, img_height = 224, 224
batch_size = 16
epochs = 100
earlystop = 10


train_generator, test_generator = data_generator(train_path, test_path, batch_size, img_height, img_width)

model = VGG16SArchitecture(0.5,img_height, img_width)
model.summary()

earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit_generator(
        train_generator,
        steps_per_epoch= 8144 // batch_size,
        epochs=100,
        validation_data = test_generator,
        validation_steps= 8041 // batch_size,
        callbacks = [earlystop])
model.save_weights('first_try.h5')

serializeModel(model, 'ConvNet' + "_initialModel")


plt.plot(history.history['val_acc'], 'r')
plt.plot(history.history['acc'], 'b')
plt.title('Performance of Standard ConvNet ')
plt.ylabel('Accuracy')
plt.xlabel('Epochs No')
plt.savefig('ConvNet_initialModel_plot.png')
plt.show()

