import tensorflow as tf
import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

batchsize = 100
num_classes = 10
epocas = 1
filas, columnas = 28, 28

(xt,yt),(xtest,ytest) = mnist.load_data()
xt = xt.reshape(xt.shape[0],filas,columnas,1)
xtest = xtest.reshape(xtest.shape[0],filas,columnas,1)

xt=xt.astype('float32')
xtest=xtest.astype('float32')

xt=xt/255
xtest=xtest/255

yt = keras.utils.to_categorical(yt,num_classes)
ytest = keras.utils.to_categorical(ytest,num_classes)

modelo = Sequential()
modelo.add(Flatten(input_shape=(28,28,1)))
modelo.add(Dense(68,activation='relu'))
modelo.add(Dense(68,activation='relu'))
modelo.add(Dense(num_classes,activation='softmax'))
modelo.summary()

modelo.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['categorical_accuracy'])

modelo.fit(xt,yt,batch_size=batchsize,epochs=epocas,verbose=1,validation_data=(xtest,ytest))

puntuacion = modelo.evaluate(xtest,ytest,batch_size=batchsize)
print(puntuacion)
