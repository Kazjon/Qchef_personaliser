import os, csv, sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.merge import Concatenate
from keras.losses import mean_squared_error
from keras.backend import round as keras_round
from user_input_training.survey_reader import survey_reader

def simpleRatingPredictor(train_xset, train_raw_ys, test_xset, test_raw_ys):
	# Initialize 4 RF predictors
	predictors_list = [RandomForestClassifier() for i in range(4)]
	# Create 1/0 arrays; 1: if > i, 0: if < i, where i is the class from 1 to 4
	train_ys_list = [[int(y > (i+1) / 5.0) for y in train_raw_ys] for i in range(4)]
	test_ys_list = [[int(y > (i+1) / 5.0) for y in test_raw_ys] for i in range(4)]
	# Iterate over the 4 predictors and fit them to training then test them
	for class_idx, (predictor, train_yset, test_yset) in enumerate(zip(predictors_list,train_ys_list,test_ys_list)):
		# Fit the predictor
		predictor.fit(train_xset,train_yset)
		# Evaluate the predictor
		print 'Predictor of class > ', class_idx + 1, ':', predictor.score(test_xset,test_yset)

def neuralRatingPredictor(train_xset, train_yset, test_xset, test_raw_ys):
	# Initialize a sequential model
	model = Sequential()
	# Create a Relu layer with the number of predictive variables
	model.add(Dense(64, activation="relu", input_dim=num_predictive_vars))
	# Add dropout layer
	model.add(Dropout(0.5))
	# Add a linear layer
	model.add(Dense(1, activation="linear"))
	# Compile with MSE as the loss measure, use RMSprop as the optimizer
	model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
	# Fit the training into the model
	model.fit(np.array(train_xset, dtype=np.float32),np.array(train_yset, dtype=np.float32),epochs=100,batch_size=100)
	# Evaluate the model using the test dataset
	# model.evaluate(np.array(test_xset, dtype=np.float32),np.array(test_yset, dtype=np.float32))
	model.evaluate(np.array(test_xset, dtype=np.float32),np.array(test_raw_ys, dtype=np.float32))

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
	print 'Current used features:', user_input_train_df.columns
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
	print 'Number of predictive variables:', num_predictive_vars
	# Create a K fold separation
	kf = KFold(n_splits=5)
	# Iterate over the splits
	for fold_idx, (train_ids, test_ids) in enumerate(kf.split(all_xs)):
		print 'fold_idx', fold_idx
		# Split the training from testing using the IDs
		train_xs, test_xs = all_xs[train_ids], all_xs[test_ids]
		train_ys, test_ys = all_ys[train_ids], all_ys[test_ids]
		# Choose and build the model
		if mode == "simple":
			simpleRatingPredictor(train_xs, train_ys, test_xs, test_ys)
		elif mode == "neural":
			neuralRatingPredictor(train_xs, train_ys, test_xs, test_ys)
			# break
