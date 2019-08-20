import numpy as np
import cv2
import os
import math
from glob import glob

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten, Input, BatchNormalization
from keras.models import Sequential, load_model, Model 
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import preprocessing

class zf_net(object):
	def __init__(self, height, width, ch, num_classes):
		self.height = height
		self.width = width
		self.ch = ch
		self.num_classes = num_classes

	def build_model(self):
		inputs = Input(shape=(self.height, self.width, self.ch))
		x = Conv2D(96, (7, 7), padding='valid', strides=2, activation='relu')(inputs)
		x = MaxPooling2D((3, 3), strides=2,  padding='same')(x)
		x = Conv2D(256, (5, 5), padding='valid', strides=2, activation='relu')(x)
		#x = keras.layers.ZeroPadding2D(1)(x)
		x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
		x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
		x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
		x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
		x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
		x = Flatten()(x)
		x = Dense(4096, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(4096, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(self.num_classes, activation='softmax')(x)

		model = Model(inputs=inputs, outputs=x, name='model')

		model.compile(loss='categorical_crossentropy',
					optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
					metrics=['accuracy'])
		model.summary()

		return model