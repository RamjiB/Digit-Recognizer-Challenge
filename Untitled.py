
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
# In[2]:

PATH ='dataset/'
# In[3]:
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH+'test.csv')
# In[4]:
def split_data(tr_data,n):
    return tr_data.iloc[:len(tr_data)-n],tr_data.iloc[len(tr_data)-n:]
# In[5]:
label = train['label']
train = train.drop('label',1)
# In[6]:
# In[7]:
num_classes = 10
# In[8]:
n = 12800
X_train,X_valid = split_data(train,n)
Y_train,Y_valid = split_data(label,n)
#one hot encoding
label1 = keras.utils.to_categorical(label, num_classes)
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_valid = keras.utils.to_categorical(Y_valid, num_classes)
# In[10]:
# input image dimensions
img_rows, img_cols = 28, 28
X_train = X_train.as_matrix()
X_valid = X_valid.as_matrix()
test = test.as_matrix()
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    test =  test.reshape(test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_valid /= 255
# test /= 255
# In[11]:
test = test.astype('float32')
test /= 255
# In[12]:
model = Sequential()
model.add(Conv2D(filters = 60, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 40, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 50, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 50, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 25, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Dropout(0.20))
model.add(Conv2D(filters = 25,kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(180, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# In[13]:
batch_size =124
epochs = 50
# In[14]:
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=RMSprop(epsilon=1e-08),
              metrics=['accuracy']) #epsilon=1e-09(previous)
# In[15]:
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)
# In[ ]:
model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_valid,Y_valid),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)
# In[ ]:
prediction = model.predict(test)
# In[ ]:
result = np.argmax(prediction,axis=1)
# In[ ]:
np.savetxt('result_DLCNN_6Con.csv',
          np.c_[range(1,len(test) + 1),result],
          delimiter=',',
            header='ImageId,Label',
           comments='',
          fmt='%d')
print("saved")
