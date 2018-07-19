import csv, sys
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

ids_row_id = [0,1]
ys_row_id = 56
simple_xs_row_ids = range(2,56)
neural_xs_row_ids = range(2,56)

def simpleRatingPredictor(train_ids, train_xs, train_raw_ys, test_ids, test_xs, test_raw_ys):
	predictors_list = [RandomForestClassifier() for i in range(4)]
	train_ys_list = [[int(y>i+1) for y in train_raw_ys] for i in range(4)]
	test_ys_list = [[int(y>i+1) for y in test_raw_ys] for i in range(4)]
	for predictor,train_ys,test_ys in zip(predictors_list,train_ys_list,test_ys_list):
		predictor.fit(train_xs,train_ys)
		print predictor.score(test_xs,test_ys)

def neuralRatingPredictor(train_ids, train_xs, train_raw_ys, test_ids, test_xs, test_raw_ys, model_filepath="best_model.h5"):
	train_xs = np.array(train_xs, dtype=np.float32)
	train_ys = np.array(train_raw_ys, dtype=np.float32)
	test_xs = np.array(test_xs, dtype=np.float32)
	test_ys = np.array(test_raw_ys, dtype=np.float32)
	
	model = Sequential()
	#layer_sizes = [64,32]
	layer_sizes = [64]
	prev_layer = len(neural_xs_row_ids)
	for layer in layer_sizes:
		model.add(Dense(layer, activation="relu",input_dim=prev_layer))
		model.add(Dropout(0.5))
		prev_layer = layer
	model.add(Dense(1, activation="linear",input_dim = prev_layer))
	model.compile(optimizer="rmsprop",loss="mse")
	callbacks = [EarlyStopping(monitor='val_loss', patience=100),
	             ModelCheckpoint(filepath=model_filepath, monitor='val_loss', save_best_only=True)]
	model.fit(train_xs,train_ys, validation_data=(test_xs,test_ys), callbacks=callbacks, epochs=10000,batch_size=100)
	best_model = load_model(model_filepath)
	print best_model.evaluate(test_xs,test_ys)



def processData(file,mode):
	with open(file, "rb") as in_f:
		reader = csv.reader(in_f)
		data = [row for row in reader]
		if type(ids_row_id) is list and len(ids_row_id) > 1:
			ids = ["".join([row[i] for i in ids_row_id]) for row in data]
		else:
			ids = [row[ids_row_id] for row in data]

		ys = [float(row[ys_row_id]) for row in data]

		if mode == "simple":
			#Hacky *5 because the simple predictor expects it.
			ys = [y*5 for y in ys]
			xs = [[float(row[id]) for id in simple_xs_row_ids] for row in data]
		elif mode == "neural":
			#Convert 1-5 to -1,0,1 by grouping 1+2 ans 4+5.  This should be probably done by creating an extra 
			new_ys = []
			for y in ys:
				if y <0.6:
					new_ys.append(-1)
				elif y == 0.6:
					new_ys.append(0)
				else:
					new_ys.append(1)
			ys = new_ys
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
