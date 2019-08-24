import numpy as np
import cv2
import os
import math
from glob import glob

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dense, Dropout, Flatten, Input, BatchNormalization
from keras.models import Sequential, load_model, Model 
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import preprocessing

class cnn(object):

	def __init__(self, height, width, ch, num_classes):
		self.height = height
		self.width = width
		self.ch = ch
		self.num_classes = num_classes

	def build_model(self):
		inputs = Input(shape=(self.height, self.width, self.ch))
		x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
		x = MaxPooling2D(pool_size=(2,2))(x)
		x = Conv2D(64, (2,2), padding='same', activation='relu')(x)
		x = MaxPooling2D(pool_size=(2,2))(x)
		#x = Flatten()(x)
		x = GlobalAveragePooling2D()(x)
		x = Dense(256, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(self.num_classes, activation='softmax')(x)

		model = Model(input=inputs, output=x)
		optimizer = Adam()
		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
		model.summary()

		return model