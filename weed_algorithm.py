#In[1]
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D

#In[2]
classifier=Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(224,224,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#In[3]
classifier.add(Flatten())

#In[4]
classifier.add(Dense(units=32,activation='relu'))
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=256,activation='relu'))

classifier.add(Dense(units=2,activation='sigmoid'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#In[5]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.15,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

#In[6]
train_path_dir="agriculture/Train"
test_path_dir="agriculture/Test"

training_set=train_datagen.flow_from_directory(train_path_dir,
                                               target_size=(224,224),
                                               batch_size=20,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory(test_path_dir,
                                          target_size=(224,224),
                                          batch_size=10,
                                          class_mode='categorical')
#In[7]
x=classifier.fit_generator(training_set,
                        steps_per_epoch=1000//20,
                        epochs=20,
                        validation_data=test_set,
                        validation_steps=60)
#In[8]
classifier.save('try_one.h5')

#In[9]
acc = x.history['acc']
val_acc =x.history['val_acc']

loss = x.history['loss']
val_loss = x.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt
plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.title("Training and validation Accuracy")
plt.figure()

plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.title("Training and validation Loss")
plt.figure()
