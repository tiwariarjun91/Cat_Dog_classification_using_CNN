import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

catDogModel= Sequential()

catDogModel.add(Conv2D(32,kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

catDogModel.add(MaxPooling2D(pool_size=(2,2)))

catDogModel.add(Conv2D(32,kernel_size=(3,3)))

catDogModel.add(Flatten())

catDogModel.add(Dense(128,activation='relu'))

catDogModel.add(Dense(1, activation='sigmoid'))

catDogModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(
        rescale= 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )
test_datagen= ImageDataGenerator(rescale=1./255)
 
train_set= train_datagen.flow_from_directory('D:\\My Stuff\\Arjun\FDP\\dataset\\training_set', target_size=(64,64),batch_size=32,class_mode='binary')
 
test_set= test_datagen.flow_from_directory('D:\\My Stuff\\Arjun\\FDP\\dataset\\test_set', target_size=(64,64), batch_size=32, class_mode='binary')
 
catDogModel.fit_generator(train_set, epochs=5, steps_per_epoch=8000, validation_data=test_set, validation_steps='none')
catDogModel.save('testModel.model')
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
newModel = load_model('testmodel.model')
test_image = image.load_img('sample1.jpg', target_size=(64,64))

test_image= image.img_to_array(test_image)

catDogModel.predict(test_image.reshape(1,64,64,3))