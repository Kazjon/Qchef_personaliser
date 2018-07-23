import os, csv, sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.merge import Concatenate
from keras.losses import mean_squared_error
from keras.backend import round as keras_round
from user_input_training.survey_reader import survey_reader

def simpleRatingPredictor(train_ids, train_xs, train_raw_ys, test_ids, test_xs, test_raw_ys):
	predictors_list = [RandomForestClassifier() for i in range(4)]
	# train_ys_list = [[int(y > i+1) for y in train_raw_ys] for i in range(4)]
	train_ys_list = [[int(y > (i+1) / 5.0) for y in train_raw_ys] for i in range(4)]
	# print 'train_ys_list', train_ys_list
	test_ys_list = [[int(y > (i+1) / 5.0) for y in test_raw_ys] for i in range(4)]
	for class_idx, (predictor, train_ys, test_ys) in enumerate(zip(predictors_list,train_ys_list,test_ys_list)):
		predictor.fit(train_xs,train_ys)
		print 'Predictor of class > ', class_idx + 1, ':', predictor.score(test_xs,test_ys)

def neuralRatingPredictor(train_ids, train_xs, train_raw_ys, test_ids, test_xs, test_raw_ys):
	model = Sequential()
	model.add(Dense(64, activation="relu", input_dim=num_predictive_vars))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation="linear"))
	model.compile(optimizer="rmsprop",loss="mse",metrics=["accuracy"])
	model.fit(np.array(train_xs, dtype=np.float32),np.array(train_ys, dtype=np.float32),epochs=100,batch_size=100)
	# model.evaluate(np.array(test_xs, dtype=np.float32),np.array(test_ys, dtype=np.float32))
	model.evaluate(np.array(test_xs, dtype=np.float32),np.array(test_raw_ys, dtype=np.float32))

if __name__ == "__main__":
	# Get the argument variables
	user_input_train_fn = sys.argv[1]
	mode = sys.argv[2]
	# Set the current working dir
	cwd = os.getcwd()
	# Get the survey object reader
	survey_reader_obj = survey_reader()
	# Input the food cuisine survey column names
	food_cuisine_survey_fn = cwd +'/personalized-surprise/Food and Cuisine Preferences Survey (Responses) - Form Responses 1'
	_, users_fam_dir_cols, _, users_cuisine_pref_cols, _, users_knowledge_cols, _, users_surp_ratings_cols, _, users_surp_pref_cols = \
		survey_reader_obj.read_survey(food_cuisine_survey_fn)
	# print users_fam_dir_cols, users_cuisine_pref_cols, users_knowledge_cols, users_surp_ratings_cols, users_surp_pref_cols
	# Read the prepared user input
	user_input_train_df = pd.read_csv(user_input_train_fn)
	# Select users or recipes
	# Remove user and recipe IDs
	user_input_train_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	# Remove unwanted features; choose to drop any of the following: users_fam_dir_cols, users_cuisine_pref_cols, users_knowledge_cols, users_surp_pref_cols
	simple_drop_cols = []
	user_input_train_df.drop(simple_drop_cols, axis=1, inplace=True)
	print user_input_train_df.columns
	# Get the target variable
	target_var = user_input_train_df['users_surp_ratings']
	all_ys = target_var.values
	# Drop the target variable
	user_input_train_df.drop(['users_surp_ratings'], axis=1, inplace=True)
	# Get predictor variables
	user_input_train_arr = user_input_train_df.values.tolist()
	all_xs = user_input_train_arr
	all_xs, all_ys = np.array(all_xs), np.array(all_ys)
	# Get number of used predictive variables
	num_predictive_vars = np.shape(all_xs)[1]
	print 'num_predictive_vars', num_predictive_vars
	# Separate train from test
	kf = KFold(n_splits=5)
	for fold_idx, (train_ids, test_ids) in enumerate(kf.split(all_xs)):
		print 'fold_idx', fold_idx
		# print("TRAIN:", train_ids, "TEST:", test_ids)
		train_xs, test_xs = all_xs[train_ids], all_xs[test_ids]
		train_ys, test_ys = all_ys[train_ids], all_ys[test_ids]
		# print 'train_ys', train_ys
		#
		if mode == "simple":
			simpleRatingPredictor(train_ids, train_xs, train_ys, test_ids, test_xs, test_ys)
		elif mode == "neural":
			neuralRatingPredictor(train_ids, train_xs, train_ys, test_ids, test_xs, test_ys)
			# break
