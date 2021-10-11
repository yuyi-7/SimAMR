from keras.layers import Conv2D, Flatten, Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

input_layer = Input(shape=(1,2,128), name='input')

conv1 = Conv2D(128, (2, 8), activation='relu', name='conv1')(input_layer)
conv1 = Dropout(0.5)(conv1)

conv2 = Conv2D(64, (1,16), activation='relu', name='conv2')(conv1)
conv2 = Dropout(0.5)(conv2)

flatten = Flatten(conv2)

dense1 = Dense(64, activation='relu', name='dense1')(flatten)
dense1 = Dropout(0.5)(dense1)

dense2 = Dense(32, activation='relu', name='dense2')(dense1)
dense2 = Dropout(0.5)(dense2)

ouput_layer = Dense(4, activation='softmax', name='output')(dense2)

model = Model(input_layer, ouput_layer)

optimizer = Adam(0.001, amsgrad=True)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

