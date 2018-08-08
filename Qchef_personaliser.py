# ToDo: Compare between the models in terms of the accuracy and MSE
	# ToDo: Try softmax layer with number of nodes == number of classes
# ToDo: Test using different features

import os, sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.merge import Concatenate
from keras.losses import mean_squared_error as keras_mean_squared_error
from keras.backend import round as keras_round
from keras.utils import to_categorical
from user_input_training.survey_reader import survey_reader

def RF_classifier(train_xset, train_raw_ys, test_xset, test_raw_ys):
	# Initialize 4 RF models
	models_list = [RandomForestClassifier() for i in range(4)]
	# Create 1/0 arrays; 1: if > i, 0: if < i, where i is the class from 1 to 4
	train_ys_list = [[int(y > (i+1)) for y in train_raw_ys] for i in range(4)]
	test_ys_list = [[int(y > (i+1)) for y in test_raw_ys] for i in range(4)]
	# train_ys_list = [[int(y > (i+1) / 5.0) for y in train_raw_ys] for i in range(4)]
	# test_ys_list = [[int(y > (i+1) / 5.0) for y in test_raw_ys] for i in range(4)]
	class_predictions_dict = {}
	models_dict = {}
	print 'Cross-validation evaluation:'
	# Iterate over the 4 models and fit them to training then test them
	for class_idx, (model, train_yset, test_yset) in enumerate(zip(models_list, train_ys_list, test_ys_list)):
		# Fit the model
		models_dict[class_idx] = model.fit(train_xset, train_yset)
		# Evaluate the model
		model_score = model.score(test_xset, test_yset)
		print 'model.score for class > ', class_idx + 1, ':', model_score
		class_prediction = model.predict(test_xset)
		# print 'Predictions of class > ', class_idx + 1, ':', class_prediction
		# Store class prediction
		class_predictions_dict[class_idx + 1] = class_prediction
	# Get final class prediction
	final_class_predictions = [0] * len(class_predictions_dict[1])
	for each_class in class_predictions_dict:
		final_class_predictions += class_predictions_dict[each_class]
	# print 'final_class_predictions', len(final_class_predictions), final_class_predictions
	# Calculate accuracy
	model_accuracy, model_accuracy_norm = accuracy_score(test_raw_ys, final_class_predictions, normalize=False), accuracy_score(test_raw_ys, final_class_predictions)
	print 'Using accuracy_score:', model_accuracy, model_accuracy_norm * 100, '%'
	model_recall_fscore = precision_recall_fscore_support(test_raw_ys, final_class_predictions, average='macro')
	print 'Using precision_recall_fscore_support:', model_recall_fscore
	return models_dict

def RF_regressor(train_xset, train_raw_ys, test_xset, test_raw_ys):
	model = RandomForestRegressor()
	model.fit(train_xset, train_raw_ys)
	model_accuracy = model.score(test_xset, test_raw_ys)
	print 'Using model.score', model_accuracy
	test_pred = model.predict(test_xset)
	# print 'Predictions:', test_pred
	test_pred = test_pred * 5
	test_pred_rounded = np.around(test_pred)
	test_raw_ys = test_raw_ys * 5
	# print 'rounding the predictions:', len(test_pred_rounded), set(test_pred_rounded), test_pred_rounded
	model_accuracy, model_accuracy_norm = accuracy_score(test_raw_ys, test_pred_rounded, normalize=False), accuracy_score(test_raw_ys, test_pred_rounded)
	print 'Using accuracy_score:', model_accuracy, model_accuracy_norm * 100, '%'
	model_recall_fscore = precision_recall_fscore_support(test_raw_ys, test_pred_rounded, average='macro')
	print 'Using precision_recall_fscore_support:', model_recall_fscore
	return model

def neuralRatingPredictor(train_xset, train_yset, test_xset, test_yset):
	# Initialize a sequential model
	model = Sequential()
	# Create a Relu layer with the number of predictive variables
	model.add(Dense(units=64, activation='relu', input_dim=num_predictive_vars))
	# Add dropout layer
	model.add(Dropout(0.5))
	# Add a linear layer
	# model.add(Dense(1, activation='linear'))
	# Add a softmax layer
	model.add(Dense(units=6, activation='softmax'))
	# Compile with MSE as the loss measure, use RMSprop as the optimizer
	model.compile(optimizer='rmsprop', loss='mse', metrics=['categorical_accuracy'])
	# model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy', 'categorical_accuracy'])
	# Determine the num_epochs and num_batches
	num_batches = 5
	num_epochs = 100
	batch_size = len(train_xset) / num_batches
	# Convert labels to categorical one-hot encoding
	hot_train_yset = to_categorical(np.array(train_yset, dtype=np.float32), num_classes=6)
	# Fit the training into the model
	# model.fit(np.array(train_xset, dtype=np.float32), np.array(train_yset, dtype=np.float32), epochs=num_epochs, batch_size=batch_size)
	model.fit(np.array(train_xset, dtype=np.float32), hot_train_yset, epochs=num_epochs, batch_size=batch_size)
	# Evaluate the model
	print 'Cross-validation evaluation:'
	# Convert labels to categorical one-hot encoding
	hot_test_yset = to_categorical(np.array(test_yset, dtype=np.float32), num_classes=6)
	# model.evaluate(np.array(test_xset, dtype=np.float32), np.array(test_yset, dtype=np.float32))
	model.evaluate(np.array(test_xset, dtype=np.float32), hot_test_yset)
	# model.evaluate(np.array(test_xset, dtype=np.float32), np.array(test_raw_ys, dtype=np.float32))
	# Predict the output of the test dataset
	# batch_size = len(test_xset) / num_batches
	test_pred = model.predict(np.array(test_xset, dtype=np.float32), batch_size=len(test_xset))
	# print 'test_pred', type(test_pred), test_pred
	test_pred_rounded = np.around(test_pred)
	# print 'rounding the predictions:', set(test_pred_rounded.flat), test_pred_rounded
	print 'test_yset', test_yset
	model_accuracy, model_accuracy_norm = accuracy_score(test_yset, test_pred_rounded, normalize=False), accuracy_score(test_yset, test_pred_rounded)
	print 'Using accuracy_score:', model_accuracy, model_accuracy_norm * 100, '%'
	model_recall_fscore = precision_recall_fscore_support(test_yset, test_pred_rounded, average='macro')
	print 'Using precision_recall_fscore_support:', model_recall_fscore
	return model

if __name__ == '__main__':
	# Get the argument variables
	user_input_fn = sys.argv[1]
	mode = sys.argv[2]
	print 'Algorithm:', mode
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
	user_input_df = pd.read_csv(user_input_fn)
	# Divide the training-validation from the test-holdout by: users, recipes or random
	msk = np.random.rand(len(user_input_df)) < 0.8
	train_df = user_input_df[msk]
	test_df = user_input_df[~msk]
	# Remove user and recipe IDs
	user_input_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	train_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	test_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	# Remove unwanted features; choose to drop any of the following: users_fam_dir_cols, users_cuisine_pref_cols, users_knowledge_cols, users_surp_pref_cols
	dropped_cols = []
	user_input_df.drop(dropped_cols, axis=1, inplace=True)
	train_df.drop(dropped_cols, axis=1, inplace=True)
	test_df.drop(dropped_cols, axis=1, inplace=True)
	print 'Current used features:', user_input_df.columns
	# Get the target variable
	target_var = user_input_df['users_surp_ratings']
	train_target_var = train_df['users_surp_ratings']
	test_target_var = test_df['users_surp_ratings']
	# Choose the model's target scale (0->1, 1->5)
	if mode == 'RF_classifier':
		target_var = target_var.values * 5
		train_target_var = train_target_var.values * 5
		test_target_var = test_target_var.values * 5
	elif mode == 'RF_regressor':
		target_var = target_var.values
		train_target_var = train_target_var.values
		test_target_var = test_target_var.values
	elif mode == 'neural':
		target_var = target_var.values * 5
		train_target_var = train_target_var.values * 5
		test_target_var = test_target_var.values * 5
	# Drop the target variable
	user_input_df.drop(['users_surp_ratings'], axis=1, inplace=True)
	train_df.drop(['users_surp_ratings'], axis=1, inplace=True)
	test_df.drop(['users_surp_ratings'], axis=1, inplace=True)
	# Get predictor variables
	predictive_var_arr = user_input_df.values.tolist()
	train_predictive_var_arr = train_df.values.tolist()
	test_predictive_var_arr = test_df.values.tolist()
	# Convert the predictive and target variables into np arrays
	predictive_var_arr, target_var = np.array(predictive_var_arr), np.array(target_var)
	train_predictive_var_arr, train_target_var = np.array(train_predictive_var_arr), np.array(train_target_var)
	test_predictive_var_arr, test_target_var = np.array(test_predictive_var_arr), np.array(test_target_var)
	# print 'test_target_var', test_target_var
	# Get number of used predictive variables
	num_predictive_vars = np.shape(predictive_var_arr)[1]
	print 'Number of predictive variables:', num_predictive_vars
	print 'Cross-validation:'
	# Create a K fold separation
	kf = KFold(n_splits=3)
	# Initialize dict of predictors to store the predictors of each fold
	predictors_dict = {}
	# Iterate over the splits
	for fold_idx, (train_ids, test_ids) in enumerate(kf.split(predictive_var_arr)):
		print 'fold_idx', fold_idx
		# Split the training from testing using the IDs
		train_xs, test_xs = predictive_var_arr[train_ids], predictive_var_arr[test_ids]
		train_ys, test_ys = target_var[train_ids], target_var[test_ids]
		# print 'test_ys', len(test_ys), set(test_ys), test_ys
		# Choose and build the model
		if mode == 'RF_classifier':
			predictors_dict[fold_idx] = RF_classifier(train_xs, train_ys, test_xs, test_ys)
		elif mode == 'RF_regressor':
			predictors_dict[fold_idx] = RF_regressor(train_xs, train_ys, test_xs, test_ys)
		elif mode == 'neural':
			predictors_dict[fold_idx] = neuralRatingPredictor(train_xs, train_ys, test_xs, test_ys)
		# break
	print 'Test predictions on test-holdout:'
	if mode ==  'RF_regressor':
		for each_predictor in predictors_dict:
			print 'each_predictor', each_predictor
			# Evaluate the model
			holdout_model_score = predictors_dict[each_predictor].score(test_predictive_var_arr, test_target_var)
			print 'Models accuracy score:', holdout_model_score * 100, '%'
			class_predictions = np.array(predictors_dict[each_predictor].predict(test_predictive_var_arr))
			# print 'Predictions of class:', class_predictions
			# Calculate accuracy
			# holdout_model_accuracy, holdout_model_accuracy_norm = accuracy_score(test_target_var, class_predictions, normalize=False), accuracy_score(test_target_var, class_predictions)
			holdout_model_accuracy = mean_squared_error(test_target_var, class_predictions)
			print 'Models mean_squared_error:', holdout_model_accuracy
			# holdout_model_recall_fscore = precision_recall_fscore_support(test_target_var, class_predictions, average='macro')
			# print 'holdout_model_recall_fscore', holdout_model_recall_fscore
	elif mode ==  'neural':
		for each_predictor in predictors_dict:
			print 'each_predictor', each_predictor
			# Evaluate the model
			predictors_dict[each_predictor].evaluate(np.array(test_predictive_var_arr, dtype=np.float32), np.array(test_target_var, dtype=np.float32))
			# Predict the output of the test dataset
			holdout_test_pred = predictors_dict[each_predictor].predict(np.array(test_predictive_var_arr, dtype=np.float32), batch_size=len(test_predictive_var_arr))
			# print 'holdout_test_pred', type(holdout_test_pred), holdout_test_pred
			holdout_test_pred_rounded = np.around(holdout_test_pred)
			# print 'rounding the predictions:', set(holdout_test_pred_rounded.flat), holdout_test_pred_rounded
			holdout_model_accuracy, holdout_model_accuracy_norm = accuracy_score(test_target_var, holdout_test_pred_rounded, normalize=False), accuracy_score(test_target_var, holdout_test_pred_rounded)
			print 'Using accuracy_score:', holdout_model_accuracy, holdout_model_accuracy_norm * 100, '%'
			holdout_model_accuracy = mean_squared_error(test_target_var, holdout_test_pred_rounded)
			print 'Models mean_squared_error:', holdout_model_accuracy
			keras_holdout_model_accuracy = keras_mean_squared_error(test_target_var, holdout_test_pred_rounded)
			print 'Models keras_holdout_model_accuracy:', keras_holdout_model_accuracy
			holdout_model_recall_fscore = precision_recall_fscore_support(test_target_var, holdout_test_pred_rounded, average='macro')
			print 'Using precision_recall_fscore_support:', holdout_model_recall_fscore
	elif mode ==  'RF_classifier':
		for each_models_dict in predictors_dict:
			print 'models_dict', each_models_dict
			models_dict = predictors_dict[each_models_dict]
			class_predictions_dict = {}
			# Iterate over the 4 models and fit them to training then test them
			for class_idx in models_dict:
				# Evaluate the model
				model_score = models_dict[class_idx].score(test_predictive_var_arr, test_target_var)
				print 'model.score for class > ', class_idx + 1, ':', model_score
				class_prediction = models_dict[class_idx].predict(test_predictive_var_arr)
				# print 'Predictions of class > ', class_idx + 1, ':', class_prediction
				# Store class prediction
				class_predictions_dict[class_idx + 1] = class_prediction
			# Get final class prediction
			final_class_predictions = [0] * len(class_predictions_dict[1])
			for each_class in class_predictions_dict:
				final_class_predictions += class_predictions_dict[each_class]
			# print 'final_class_predictions', len(final_class_predictions), final_class_predictions
			# Calculate accuracy
			holdout_model_accuracy, holdout_model_accuracy_norm = accuracy_score(test_target_var, final_class_predictions, normalize=False), accuracy_score(test_target_var, final_class_predictions)
			print 'Using accuracy_score:', holdout_model_accuracy, holdout_model_accuracy_norm * 100, '%'
			holdout_model_accuracy = mean_squared_error(test_target_var, final_class_predictions)
			print 'Models mean_squared_error:', holdout_model_accuracy
			holdout_model_recall_fscore = precision_recall_fscore_support(test_target_var, final_class_predictions, average='macro')
			print 'Using precision_recall_fscore_support:', holdout_model_recall_fscore
