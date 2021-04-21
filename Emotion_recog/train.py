import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.models import Sequential as sq
from keras.layers import Dense, Activation as act, Flatten, BatchNormalization as bn, Conv2D, MaxPooling2D as mp2d, Dropout
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint as mdc, EarlyStopping as es, ReduceLROnPlateau as rlp

class_num = 7
img = dict()
img['row'] = 48
img['col'] = 48
batch_size = 16

train_data_loc = "./images/train"
validation_data_loc = "./images/validation"
train_sample_number = 28821
validation_sample_number = 7066
# Preprocess and Generate the training data from the train images we already have
training_data = idg(rescale=1./255, rotation_range=30, zoom_range=0.4,
                    width_shift_range=0.3, height_shift_range=0.3, shear_range=0.4, horizontal_flip=True, fill_mode='nearest')
# Preprocess validation data
validation_data = idg(rescale=1./255)

# Fit the data
training_parameter = training_data.flow_from_directory(train_data_loc, color_mode='grayscale', target_size=(
    img['row'], img['col']), batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_parameter = validation_data.flow_from_directory(train_data_loc, color_mode='grayscale', target_size=(
    img['row'], img['col']), batch_size=batch_size, class_mode='categorical', shuffle=True)

print(training_data)
print(validation_data)
# Define a sequential model
model = sq()

# Layer1
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                 input_shape=(img['row'], img['col'], 1)))
model.add(act('elu'))
model.add(bn())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                 input_shape=(img['row'], img['col'], 1)))
model.add(act('elu'))
model.add(bn())
model.add(mp2d(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Layer2
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(mp2d(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Layer3
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(mp2d(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Layer4
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(mp2d(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Layer5
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(Dropout(0.5))

# Layer6
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(act('elu'))
model.add(bn())
model.add(Dropout(0.5))

# Layer 7
model.add(Dense(class_num, kernel_initializer='he_normal'))
model.add(act('softmax'))

print(model.summary())

early_stop = es(monitor='val_loss', min_delta=0, patience=3,
                verbose=1, restore_best_weights=True)

check_point = mdc('./checkpoint.h5', monitor='val_loss',
                  mode='min', save_best_only=True, verbose=1)

reduce_learning_rate = rlp(monitor='val_loss', patience=3,
                           verbose=1, factor=0.2, min_delta=0.0001)

callback = [early_stop, check_point, reduce_learning_rate]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001), metrics=['accuracy'])

epochs = 50

history = model.fit_generator(
    training_parameter, steps_per_epoch=train_sample_number//batch_size,
    epochs=epochs, callbacks=callback, validation_data=validation_parameter,
    validation_steps=validation_sample_number//batch_size)
