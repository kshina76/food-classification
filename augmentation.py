import gc
import os
import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
from skimage import transform
from skimage import util


class dataAugmentation():
	def __init__(self, directory, height, width):
		self.directory = directory + '/aug'
		self.height = height
		self.width = width

	def makeDir(self):
		if not os.path.isdir(self.directory):
			os.makedirs(self.directory)

	def loadData(self):
		for i in range(self.data_path):
			img = load_img(self.data_path[i], target_size=(self.height, self.width))
			self.flipHorizontal(img, i)
			self.rotateImage(img, i)
			self.noisyImage(img, i)

		del img
		gc.collect()

	# 左右反転
	def flipHorizontal(self, img, index):
		save_img(self.directory + str(index), img[:, ::-1, :])

	# 回転
	def rotateImage(self, img, index):
		img = transform.rotate(img, angle=random.randint(-15, 15), resize=False, center=None)
		save_img(self.directory + str(index), img)

	# ノイズ
	def noisyImage(self, img, index):
		img = util.random_noise(img)
		img = img.fromarray((img*255).astype(np.uint8))
		save_img(self.directory + str(index), img)