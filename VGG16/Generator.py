import numpy as np
import math

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import Sequence
from keras.utils import np_utils
from sklearn import preprocessing

class BatchGenerator(Sequence):

	def __init__(self, data_path, labels, batch_size, height, width, ch=3):
		self.batch_size = batch_size
		self.data_path = data_path
		self.labels = labels
		self.length = len(data_path)
		self.height = height
		self.width = width
		self.ch = ch
		self.batches_per_epoch = math.ceil(self.length / batch_size) #バッチの個数

	def __getitem__(self, idx):
		# 1つのミニバッチ分のデータ範囲
		batch_from = self.batch_size * idx
		batch_to = batch_from + self.batch_size

		# バッチ処理の最後の方バッチサイズ分のデータが足りなくなった時の処理
		if batch_to > self.length:
			batch_to = self.length

		x_batch = []  # feature
		y_batch = []  # target

		for i in range(batch_from, batch_to):
			img = load_img(self.data_path[i], target_size=(self.height, self.width))
			img_array = img_to_array(img) / 255  # normalization
			x_batch.append(img_array)
			y_batch.append(self.labels[i])

		# appendでリストになっているから、配列にしておく
		x_batch = np.asarray(x_batch)
		y_batch = np.asarray(y_batch)

		#modelの入力が、(batch_size, height, width, channel)
		#出力が、(batch_size, labels) なので、しっかりbatch_sizeの部分を実装してあげる
		#shape -> x = (64, 227, 227, 3) y = (64, 101)
		return x_batch, y_batch

	def __len__(self):
		return self.batches_per_epoch


	def on_epoch_end(self):
		pass