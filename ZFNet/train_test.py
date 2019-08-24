import numpy as np
import os
import argparse
import Generator
import ZFNet

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

directory = '../food-3/images'
class_path = '../food-3/meta/classes.txt'
save_model_name = 'ZFNet.h5'
load_model_name = 'ZFNet.h5'
height = 224
width = 224
ch = 3
batch_size = 64
epoch = 50

def define_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-mode', help='select mode : train or test', required=True)
	args = parser.parse_args()

	return args

def train_test():

	args = define_parse()

	# make labels and paths
	data_path = []
	labels = []
	label = -1
	for dir_path, dir_name, file_name in os.walk(directory):
		for file_name in file_name:
			data_path.append(os.path.join(dir_path, file_name))
			labels.append(label)
		label += 1

	# transform labels into one-hot-vector
	labels_onehot = to_categorical(labels)
	print(labels_onehot.shape)

	#何クラス分類なのか。今回は101クラス分類なので101が入る
	num_classes = label

	# split data to training and test data
	X_train, X_test, y_train, y_test = train_test_split(data_path, labels_onehot, train_size=0.8)

	# for making validation data
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8)

	# make Generator for fit_generator
	train_batch_generator = Generator.BatchGenerator(X_train, y_train, batch_size, height, width)
	test_batch_generator = Generator.BatchGenerator(X_val, y_val, batch_size, height, width)

	if args.mode == 'train':
		ZF = ZFNet.zf_net(height, width, ch, num_classes)

		model = ZF.build_model()

		# training
		fit_history = model.fit_generator(train_batch_generator, epochs=epoch, verbose=1,
											steps_per_epoch=train_batch_generator.batches_per_epoch,
											validation_data=test_batch_generator,
											validation_steps=test_batch_generator.batches_per_epoch,
											shuffle=True
											)
	
		model.save(save_model_name)

		# evaluate
		'''
		score = model.evaluate_generator(test_batch_generator,
										step=train_batch_generator.batches_per_epoch,
										verbose=1)
		'''

	elif args.mode == 'test':
		model = load_model(load_model_name)

		# get class name for predicting
		class_name = []
		with open(class_path, "r") as file:
			for i in file:
				class_name.append(i.replace('\n', ''))
		class_name = np.asarray(class_name)

		# prediction
		img = load_img(X_test[0], target_size=(height, width))
		img_array = img_to_array(img) / 255  # normalization
		img_array = np.expand_dims(img_array, axis=0)  #add dimention that is batch_size

		pred = model.predict(img_array, verbose=0)

		print('prediction result : {}'.format(class_name[np.argmax(pred[0,:])]))
		print('correct answer : {}'.format(class_name[np.argmax(y_test[0,:])]))

	else:
		print('illegal input.')
		print('please select train or test')


train_test()