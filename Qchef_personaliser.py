import csv, sys
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.merge import Concatenate
from keras.losses import mean_squared_error
from keras.backend import round as keras_round

import numpy as np

ids_row_id = 0
ys_row_id = 1
simple_xs_row_ids = [1]
neural_xs_row_ids = [1,2,3]

def simpleRatingPredictor(train_ids, train_xs, train_raw_ys, test_ids, test_xs, test_raw_ys):
	predictors_list = [RandomForestClassifier() for i in range(4)]
	train_ys_list = [[int(y>i+1) for y in train_raw_ys] for i in range(4)]
	test_ys_list = [[int(y>i+1) for y in test_raw_ys] for i in range(4)]
	for predictor,train_ys,test_ys in zip(predictors_list,train_ys_list,test_ys_list):
		predictor.fit(train_xs,train_ys)
		print predictor.score(test_xs,test_ys)

def neuralRatingPredictor(train_ids, train_xs, train_raw_ys, test_ids, test_xs, test_raw_ys):
	model = Sequential()
	model.add(Dense(64, activation="relu",input_dim=len(neural_xs_row_ids)))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation="linear"))
	model.compile(optimizer="rmsprop",loss="mse",metrics=["accuracy"])
	model.fit(np.array(train_xs, dtype=np.float32),np.array(train_ys, dtype=np.float32),epochs=100,batch_size=100)
	model.evaluate(np.array(test_xs, dtype=np.float32),np.array(test_ys, dtype=np.float32))



def processData(file,mode):
	with open(file, "rb") as in_f:
		reader = csv.reader(in_f)
		data = [row for row in reader]
		ids = [row[ids_row_id] for row in data]
		ys = [int(row[ys_row_id]) for row in data]
		if mode == "simple":
			xs = [[float(row[id]) for id in simple_xs_row_ids] for row in data]
		elif mode == "neural":
			xs = [[float(row[id]) for id in neural_xs_row_ids] for row in data]
		else:
			raise NotImplementedError
	return ids,xs,ys


if __name__ == "__main__":
	mode = sys.argv[3]
	train_ids,train_xs,train_ys = processData(sys.argv[1],mode)
	test_ids,test_xs,test_ys = processData(sys.argv[2],mode)
	if mode == "simple":
		simpleRatingPredictor(train_ids, train_xs, train_ys, test_ids, test_xs, test_ys)
	elif mode == "neural":
		neuralRatingPredictor(train_ids, train_xs, train_ys, test_ids, test_xs, test_ys)
