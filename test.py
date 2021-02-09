from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image 


datagen = ImageDataGenerator()

train = datagen.flow_from_directory('/CHEMINAMODIFIER/whale-categorization-playground/data/', class_mode='categorical', batch_size=64, target_size=(256, 256))
print(train[0])

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4251))
model.add(Activation('softmax'))

#model.add(Dense(10, name='fc1'))
#model.add(Activation('softmax'))

learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

nb_epoch = 1
model.fit(train, epochs=nb_epoch, verbose=1) #train_it,batch_size=batch_size, epochs=nb_epoch,verbose=1

#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))


#batch_size = 32
#img_height = 180
#img_width = 180

#train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#  '/Users/priscille/Desktop/whale-categorization-playground/data/',
#  validation_split=0.2,
#  subset="training",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size=batch_size)