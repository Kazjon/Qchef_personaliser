import os, csv, sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.merge import Concatenate
from keras.losses import mean_squared_error
from keras.backend import round as keras_round
from user_input_training.survey_reader import survey_reader

def RF_classifier(train_xset, train_raw_ys, test_xset, test_raw_ys):
	# Initialize 4 RF predictors
	predictors_list = [RandomForestClassifier() for i in range(4)]
	# Create 1/0 arrays; 1: if > i, 0: if < i, where i is the class from 1 to 4
	train_ys_list = [[int(y > (i+1)) for y in train_raw_ys] for i in range(4)]
	test_ys_list = [[int(y > (i+1)) for y in test_raw_ys] for i in range(4)]
	# train_ys_list = [[int(y > (i+1) / 5.0) for y in train_raw_ys] for i in range(4)]
	# test_ys_list = [[int(y > (i+1) / 5.0) for y in test_raw_ys] for i in range(4)]
	class_predictions_dict = {}
	# Iterate over the 4 predictors and fit them to training then test them
	for class_idx, (predictor, train_yset, test_yset) in enumerate(zip(predictors_list, train_ys_list, test_ys_list)):
		# Fit the predictor
		predictor.fit(train_xset, train_yset)
		# Evaluate the predictor
		predictor_score = predictor.score(test_xset, test_yset)
		print 'Predictors accuracy of class > ', class_idx + 1, ':', predictor_score
		class_prediction = predictor.predict(test_xset)
		# print 'Predictions of class > ', class_idx + 1, ':', class_prediction
		# Store class prediction
		class_predictions_dict[class_idx + 1] = class_prediction
	# Get final class prediction
	final_class_predictions = [0] * len(class_predictions_dict[1])
	for each_class in class_predictions_dict:
		final_class_predictions += class_predictions_dict[each_class]
	# print 'final_class_predictions', len(final_class_predictions), final_class_predictions
	# Calculate accuracy
	predictor_accuracy, predictor_accuracy_norm = accuracy_score(test_raw_ys, final_class_predictions, normalize=False), accuracy_score(test_raw_ys, final_class_predictions)
	print 'predictor_accuracy', predictor_accuracy, predictor_accuracy_norm * 100, '%'
	predictor_recall_fscore = precision_recall_fscore_support(test_raw_ys, final_class_predictions, average='macro')
	print 'predictor_recall_fscore', predictor_recall_fscore

def neuralRatingPredictor(train_xset, train_yset, test_xset, test_yset):
	# Initialize a sequential model
	model = Sequential()
	# Create a Relu layer with the number of predictive variables
	model.add(Dense(64, activation='relu', input_dim=num_predictive_vars))
	# Add dropout layer
	model.add(Dropout(0.5))
	# Add a linear layer
	model.add(Dense(1, activation='linear'))
	# Compile with MSE as the loss measure, use RMSprop as the optimizer
	model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy', 'categorical_accuracy'])
	# Determine the num_epochs and num_batches
	num_batches = 20
	num_epochs = 10000
	batch_size = len(train_xset) / num_batches
	# Fit the training into the model
	# model.fit(np.array(train_xset, dtype=np.float32), np.array(train_yset, dtype=np.float32), epochs=num_epochs, batch_size=100)
	# model.fit(np.array(train_xset, dtype=np.float32), np.array(train_yset, dtype=np.float32), epochs=num_epochs, batch_size=len(test_xset))
	# model.fit(np.array(train_xset, dtype=np.float32), np.array(train_yset, dtype=np.float32), epochs=num_epochs, batch_size=len(train_xset))
	model.fit(np.array(train_xset, dtype=np.float32), np.array(train_yset, dtype=np.float32), epochs=num_epochs, batch_size=batch_size)
	# Evaluate the model using the test dataset
	print 'Evaluation:'
	model.evaluate(np.array(test_xset, dtype=np.float32), np.array(test_yset, dtype=np.float32))
	# model.evaluate(np.array(test_xset, dtype=np.float32), np.array(test_raw_ys, dtype=np.float32))
	# Predict the output of the test dataset
	test_pred = model.predict(np.array(test_xset, dtype=np.float32), batch_size=len(test_xset))
	# print 'test_pred', type(test_pred), test_pred
	test_pred_rounded = np.around(test_pred)
	# print 'rounding the predictions:', set(test_pred_rounded.flat), test_pred_rounded
	predictor_accuracy, predictor_accuracy_norm = accuracy_score(test_yset, test_pred_rounded, normalize=False), accuracy_score(test_yset, test_pred_rounded)
	print 'predictor_accuracy', predictor_accuracy, predictor_accuracy_norm * 100, '%'
	predictor_recall_fscore = precision_recall_fscore_support(test_yset, test_pred_rounded, average='macro')
	print 'predictor_recall_fscore', predictor_recall_fscore

if __name__ == '__main__':
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
	# Divide users or recipes
	# Remove user and recipe IDs
	user_input_train_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	# Remove unwanted features; choose to drop any of the following: users_fam_dir_cols, users_cuisine_pref_cols, users_knowledge_cols, users_surp_pref_cols
	simple_drop_cols = []
	user_input_train_df.drop(simple_drop_cols, axis=1, inplace=True)
	print 'Current used features:', user_input_train_df.columns
	# Get the target variable
	target_var = user_input_train_df['users_surp_ratings']
	all_ys = []
	# Choose and build the model
	if mode == 'RF_classifier':
		all_ys = target_var.values * 5
	elif mode == 'RF_regressor':
		all_ys = target_var.values
	elif mode == 'neural':
		all_ys = target_var.values * 5
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
		# print 'test_ys', len(test_ys), set(test_ys), test_ys
		# Choose and build the model
		if mode == 'RF_classifier':
			RF_classifier(train_xs, train_ys, test_xs, test_ys)
		elif mode == 'RF_regressor':
			RF_regressor(train_xs, train_ys, test_xs, test_ys)
		elif mode == 'neural':
			neuralRatingPredictor(train_xs, train_ys, test_xs, test_ys)
		break
