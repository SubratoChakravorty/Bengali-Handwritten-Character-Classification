import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import time, gc
import cv2
import albumentations as A
from PIL import Image
from tensorflow import keras
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model, load_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.io as io
from keras import backend as K
from .custom_generator import MultiOutputDataGenerator
fname = 'densenet'



from keras.applications import DenseNet121
from keras.layers import GlobalAveragePooling2D

IMG_SIZE = 64
N_CHANNELS = 1

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

from keras.layers import LeakyReLU, Add, MaxPooling2D

densenet = DenseNet121(weights='imagenet', include_top=False)

x = Conv2D(3, (3, 3), padding='same')(inputs)

x = densenet(x)

x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
dense = Dropout(0.5)(x)

# model = Dense(1024, activation = "relu")(x)
# model = Dropout(rate=0.3)(model)
# dense = Dense(512, activation = "relu")(x)

head_root = Dense(168, activation='softmax', name='dense_3')(dense)
head_vowel = Dense(11, activation='softmax', name='dense_4')(dense)
head_consonant = Dense(7, activation='softmax', name='dense_5')(dense)


model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
model.summary()
# filepath = "weights_best_val_acc_" + fname + ".hdf5"
# modelcheckpoint = ModelCheckpoint(filepath, monitor='val_dense_2_accuracy', save_best_only=True, verbose=1)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
# filepath="weights_best_val_loss_"+fname+".hdf5"
# modelcheckpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True, verbose=1)

batch_size = 256
epochs = 20
histories = []
hist_mat = []

train_df = pd.read_csv('./input/train_df.csv')
val_df = pd.read_csv('./input/val_df.csv')

datagen = MultiOutputDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.15,  # Randomly zoom image
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,
    rescale=1. / 255.)

val_datagen = MultiOutputDataGenerator(rescale=1. / 255.)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="./input/png_images",
    x_col="filepath",
    y_col=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'],
    batch_size=batch_size,
    color_mode='grayscale',
    seed=42,
    shuffle=True,
    class_mode="multi_output",
    target_size=(64, 64), keys=['dense_2', 'dense_3', 'dense_4'])

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory="./input/png_images",
    x_col="filepath",
    y_col=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'],
    batch_size=batch_size,
    color_mode='grayscale',
    seed=42,
    shuffle=True,
    class_mode="multi_output",
    target_size=(64, 64), keys=['dense_2', 'dense_3', 'dense_4'])

history = model.fit_generator(generator=train_generator,
                              epochs=epochs, validation_data=val_generator,
                              steps_per_epoch=len(train_df) // batch_size,
                              validation_steps=len(val_df) // batch_size)

history_df = pd.DataFrame(history.history)
history_df.to_csv('../histories/history_densenet.csv')
