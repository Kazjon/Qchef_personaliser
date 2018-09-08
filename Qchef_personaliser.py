# ToDo: Compare between the models in terms of the accuracy and MSE
	# ToDo: Try softmax layer with number of nodes == number of classes
# ToDo: Test using different features
# ToDo: Make switching between using the target variable scaled before or after training optinal through arguments
# ToDo: Make switching between categorical and continuous for the NNW optinal through arguments

import os, sys, math, statistics, random, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, precision_recall_fscore_support, confusion_matrix
from GloVex.evaluate_personalised import survey_reader

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
	# test_pred = test_pred * 5
	test_pred_rounded = np.around(test_pred)
	# test_raw_ys = test_raw_ys * 5
	# print 'rounding the predictions:', len(test_pred_rounded), set(test_pred_rounded), test_pred_rounded
	model_accuracy, model_accuracy_norm = accuracy_score(test_raw_ys, test_pred_rounded, normalize=False), accuracy_score(test_raw_ys, test_pred_rounded)
	print 'Using rounded for accuracy_score:', model_accuracy, model_accuracy_norm * 100, '%'
	# model_recall_fscore = precision_recall_fscore_support(test_raw_ys, test_pred, average='macro')
	# print 'Using precision_recall_fscore_support:', model_recall_fscore
	model_recall_fscore = precision_recall_fscore_support(test_raw_ys, test_pred_rounded, average='macro')
	print 'Using rounded for precision_recall_fscore_support:', model_recall_fscore
	return model

def neuralRatingPredictor(train_xset, train_yset, test_xset, test_yset):
	# Initialize a sequential model
	model = Sequential()
	# Create a Relu layer with the number of predictive variables
	model.add(Dense(units=64, activation='relu', input_dim=num_predictive_vars))
	# Add dropout layer
	model.add(Dropout(0.5))
	# Add a linear layer
	model.add(Dense(1, activation='linear'))
	# Add a softmax layer
	# model.add(Dense(units=5, activation='softmax'))
	# Compile with MSE as the loss measure, use RMSprop as the optimizer
	# model.compile(optimizer='rmsprop', loss='mse', metrics=['categorical_accuracy'])
	model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy', 'categorical_accuracy'])
	# Determine the num_epochs and num_batches
	num_batches = 5
	num_epochs = 1000
	batch_size = len(train_xset) / num_batches
	# Convert labels to categorical one-hot encoding
	# hot_train_yset = to_categorical(np.array(train_yset, dtype=np.float32), num_classes=6)
	# Fit the training into the model
	model.fit(np.array(train_xset, dtype=np.float32), np.array(train_yset, dtype=np.float32), epochs=num_epochs, batch_size=batch_size)
	# model.fit(np.array(train_xset, dtype=np.float32), hot_train_yset, epochs=num_epochs, batch_size=batch_size)
	# Evaluate the model
	print 'Cross-validation evaluation:'
	# Convert labels to categorical one-hot encoding
	# hot_test_yset = to_categorical(np.array(test_yset, dtype=np.float32), num_classes=6)
	model.evaluate(np.array(test_xset, dtype=np.float32), np.array(test_yset, dtype=np.float32))
	# model.evaluate(np.array(test_xset, dtype=np.float32), hot_test_yset)
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

def within_one_accuracy_fn(__confusion_matrix_arr__):
	__within_one_accuracy__ = 0
	for each_label in range(len(__confusion_matrix_arr__) - 1):
		__within_one_accuracy__ += __confusion_matrix_arr__[each_label][each_label] + \
								   __confusion_matrix_arr__[each_label][each_label + 1] + \
								   __confusion_matrix_arr__[each_label + 1][each_label]
	__within_one_accuracy__ += __confusion_matrix_arr__[-1][-1]
	return __within_one_accuracy__

def get_recipes_cuisine(row):
	b = (user_input_df[cuisine_softmax_cols].ix[row.name] == 0.2)
	return b.idxmax().replace('_softmax', '')

def get_attr_cuisine(row, attr):
	selected_cuisine = row['recipes_cuisine'] + attr
	if selected_cuisine == 'modern_cuisine_pref':
		average_cuisine_pref = sum(list(user_input_df[cuisine_preference_cols].ix[row.name])) / float(len(cuisine_preference_cols))
		return average_cuisine_pref
	return user_input_df[selected_cuisine].ix[row.name]

if __name__ == '__main__':
	# Get the argument variables
	user_input_fn = sys.argv[1]
	algo_mode = sys.argv[2]
	food_cuisine_survey_fp = sys.argv[3]
	print 'Algorithm:', algo_mode
	# Load keras libraries if using NNWs (to save time when using other techniques)
	if algo_mode == 'neural':
		from keras.models import Sequential
		from keras.layers import Dense, Dropout, Activation
		from keras.layers.merge import Concatenate
		from keras.losses import mean_squared_error as keras_mean_squared_error
		from keras.backend import round as keras_round
		from keras.utils import to_categorical
	# Set the current working dir
	cwd = os.getcwd()
	# Get the survey object reader
	survey_reader_obj = survey_reader()
	# Input the food cuisine survey column names
	food_cuisine_survey_fn = cwd + food_cuisine_survey_fp
	# _, users_fam_cols, _, users_cuisine_pref_cols, _, users_knowledge_cols, _, users_surp_ratings_cols, _, users_surp_pref_cols = \
	# 	survey_reader_obj.read_survey(food_cuisine_survey_fn)
	fam_cat_sorted = ['mexican', 'chinese', 'modern', 'greek', 'indian', 'thai', 'italian']
	_, users_fam_cols, _, users_cuisine_pref_cols, \
	_, knowledge_ingredient_cols, _, cuisine_knowledge_cols, \
	_, surprise_rating_cols, _, users_surp_pref_cols = \
		survey_reader_obj.read_survey(food_cuisine_survey_fn, fam_cat_sorted)
	cuisine_softmax_cols = [each_cuisine + '_softmax' for each_cuisine in fam_cat_sorted]
	cuisine_preference_cols = [_each_ + '_cuisine_pref' for _each_ in fam_cat_sorted if _each_ != 'modern']
	cuisine_fam_dir = [_each_ + '_fam_dir' for _each_ in fam_cat_sorted]
	# Read the prepared user input
	user_input_df = pd.read_csv(user_input_fn)
	print 'Number of records:', len(user_input_df)
	print 'Unique surprise ratings:', user_input_df['users_surp_ratings'].unique()
	print 'Number of unique surprise ratings:', user_input_df['Recipe ID'].nunique()
	# Get the familiarity, knowledge and preference of the recipe's cuisine
	user_input_df['recipes_cuisine'] = user_input_df[cuisine_softmax_cols].apply(get_recipes_cuisine, axis=1)
	print 'user_input_df', user_input_df.columns
	user_input_df['recipe_familiarity'] = user_input_df.apply(lambda x: get_attr_cuisine(x, attr='_fam_dir'), axis=1)
	user_input_df['recipe_knowledge'] = user_input_df.apply(lambda x: get_attr_cuisine(x, attr='_cuisine_knowledge'), axis=1)
	user_input_df['recipe_preference'] = user_input_df.apply(lambda x: get_attr_cuisine(x, attr='_cuisine_pref'), axis=1)

	# Remove unsure records
	user_input_df = user_input_df[user_input_df['users_surp_ratings'] != -0.2]
	print 'After filtering out the unsure records:'
	print 'Number of records:', len(user_input_df)
	print 'Unique surprise ratings:', user_input_df['users_surp_ratings'].unique()
	print 'Number of unique surprise ratings:', user_input_df['Recipe ID'].nunique()
	# Divide the training-validation from the test-holdout by: users, recipes or random
	msk = np.random.rand(len(user_input_df)) < 0.8
	train_df = user_input_df[msk]
	test_df = user_input_df[~msk]
	# Remove user and recipe IDs
	user_input_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	train_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	test_df.drop(['Recipe ID', 'User ID'], axis=1, inplace=True)
	# Remove unwanted features; choose to drop any of the following:
	# Lists of column names: users_fam_cols + users_cuisine_pref_cols + cuisine_knowledge_cols + cuisine_softmax_cols + \
	# Individual column names:
	# ['observed_surp_estimates_90perc', 'observed_surp_estimates_95perc', 'observed_surp_estimates_max',
	#  'oracle_surp_estimates_90perc', 'oracle_surp_estimates_95perc', 'oracle_surp_estimates_max',
	#  'personalized_surp_estimates_90perc', 'personalized_surp_estimates_95perc', 'personalized_surp_estimates_max',
	#  'users_surp_pref'
	#  'recipe_familiarity', 'recipe_knowledge', 'recipe_preference']
	dropped_cols = users_fam_cols + users_cuisine_pref_cols + cuisine_knowledge_cols + cuisine_softmax_cols + \
	[
	 # 'observed_surp_estimates_90perc', 'observed_surp_estimates_95perc', 'observed_surp_estimates_max',
	 'oracle_surp_estimates_90perc', 'oracle_surp_estimates_95perc', 'oracle_surp_estimates_max',
	 'personalized_surp_estimates_90perc', 'personalized_surp_estimates_95perc', 'personalized_surp_estimates_max',
	 'users_surp_pref',
	 # 'recipe_familiarity',
	 # 'recipe_knowledge',
	 # 'recipe_preference',
	 'recipes_cuisine']
	user_input_df.drop(dropped_cols, axis=1, inplace=True)
	train_df.drop(dropped_cols, axis=1, inplace=True)
	test_df.drop(dropped_cols, axis=1, inplace=True)
	print 'Current used features:', user_input_df.columns
	# Get the target variable
	target_var = user_input_df['users_surp_ratings']
	train_target_var = train_df['users_surp_ratings']
	test_target_var = test_df['users_surp_ratings']
	# Choose the model's target scale (0->1, 1->5)
	if algo_mode == 'RF_classifier':
		target_var = target_var.values * 5
		train_target_var = train_target_var.values * 5
		test_target_var = test_target_var.values * 5
	elif algo_mode == 'RF_regressor':
		target_var = target_var.values * 5
		train_target_var = train_target_var.values * 5
		test_target_var = test_target_var.values * 5
	elif algo_mode == 'neural':
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
	# Repeat the experiment num_exp times
	num_exp = 10
	within_one_accuracy_perc_arr = []
	for experiment_i in range(num_exp):
		print 'experiment number:', experiment_i
		print 'Cross-validation:'
		# Create a K fold separation
		kf = KFold(n_splits=3, shuffle=True)
		# Initialize dict of predictors to store the predictors of each fold
		predictors_dict = {}
		# Iterate over the splits
		for fold_idx, (train_ids, test_ids) in enumerate(kf.split(train_predictive_var_arr)):
			print 'fold_idx', fold_idx
			# Split the training from validation (not testing; that's why we are using the training set) using the IDs
			train_xs, test_xs = train_predictive_var_arr[train_ids], train_predictive_var_arr[test_ids]
			train_ys, test_ys = train_target_var[train_ids], train_target_var[test_ids]
			# Choose and build the model
			if algo_mode == 'RF_classifier':
				predictors_dict[fold_idx] = RF_classifier(train_xs, train_ys, test_xs, test_ys)
			elif algo_mode == 'RF_regressor':
				predictors_dict[fold_idx] = RF_regressor(train_xs, train_ys, test_xs, test_ys)
			elif algo_mode == 'neural':
				predictors_dict[fold_idx] = neuralRatingPredictor(train_xs, train_ys, test_xs, test_ys)
			# break
		print 'Test predictions on test-holdout:'
		if algo_mode ==  'RF_regressor':
			for each_predictor in predictors_dict:
				print 'each_predictor', each_predictor
				# Evaluate the model
				holdout_model_score = predictors_dict[each_predictor].score(test_predictive_var_arr, test_target_var)
				print 'Models accuracy score:', holdout_model_score * 100, '%'
				class_predictions = np.array(predictors_dict[each_predictor].predict(test_predictive_var_arr))
				print 'class_predictions', class_predictions
				class_predictions_rounded = np.around(class_predictions)
				print 'class_predictions_rounded', class_predictions_rounded
				# Calculate accuracy
				holdout_model_accuracy, holdout_model_accuracy_norm = accuracy_score(test_target_var, class_predictions_rounded, normalize=False), accuracy_score(test_target_var, class_predictions_rounded)
				print 'Using accuracy_score:', holdout_model_accuracy, holdout_model_accuracy_norm * 100, '%'
				holdout_model_accuracy = mean_squared_error(test_target_var, class_predictions)
				print 'Models mean_squared_error:', holdout_model_accuracy
				holdout_model_accuracy = mean_absolute_error(test_target_var, class_predictions)
				print 'Models mean_absolute_error:', holdout_model_accuracy
				holdout_model_recall_fscore = precision_recall_fscore_support(test_target_var, class_predictions_rounded, average='macro')
				print 'holdout_model_recall_fscore', holdout_model_recall_fscore
				confusion_matrix_arr = confusion_matrix(test_target_var, class_predictions_rounded, labels=list(set(test_target_var)))
				print 'Confusion matrix:\n', confusion_matrix_arr
				within_one_accuracy = within_one_accuracy_fn(confusion_matrix_arr)
				within_one_accuracy_perc = within_one_accuracy / float(len(test_target_var)) * 100
				print 'within_one_accuracy', within_one_accuracy, within_one_accuracy_perc, '%'
				within_one_accuracy_perc_arr.append(within_one_accuracy_perc)

		elif algo_mode ==  'neural':
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
				holdout_model_accuracy = mean_squared_error(test_target_var, holdout_test_pred)
				print 'Models mean_squared_error:', holdout_model_accuracy
				holdout_model_accuracy = mean_absolute_error(test_target_var, holdout_test_pred)
				print 'Models mean_absolute_error:', holdout_model_accuracy
				keras_holdout_model_accuracy = keras_mean_squared_error(test_target_var, holdout_test_pred)
				print 'Models keras_holdout_model_accuracy:', keras_holdout_model_accuracy
				holdout_model_recall_fscore = precision_recall_fscore_support(test_target_var, holdout_test_pred_rounded, average='macro')
				print 'Using precision_recall_fscore_support:', holdout_model_recall_fscore
				confusion_matrix_arr = confusion_matrix(test_target_var, holdout_test_pred_rounded, labels=list(set(test_target_var)))
				print 'Confusion matrix:\n', confusion_matrix_arr
				within_one_accuracy = within_one_accuracy_fn(confusion_matrix_arr)
				within_one_accuracy_perc = within_one_accuracy / float(len(test_target_var)) * 100
				print 'within_one_accuracy', within_one_accuracy, within_one_accuracy_perc, '%'
				within_one_accuracy_perc_arr.append(within_one_accuracy_perc)

		elif algo_mode ==  'RF_classifier':
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
				holdout_model_accuracy = mean_absolute_error(test_target_var, final_class_predictions)
				print 'Models mean_absolute_error:', holdout_model_accuracy
				holdout_model_recall_fscore = precision_recall_fscore_support(test_target_var, final_class_predictions, average='macro')
				print 'Using precision_recall_fscore_support:', holdout_model_recall_fscore
				confusion_matrix_arr = confusion_matrix(test_target_var, final_class_predictions, labels=list(set(test_target_var)))
				print 'Confusion matrix:\n', confusion_matrix_arr
				within_one_accuracy = within_one_accuracy_fn(confusion_matrix_arr)
				within_one_accuracy_perc = within_one_accuracy / float(len(test_target_var)) * 100
				print 'within_one_accuracy', within_one_accuracy, within_one_accuracy_perc, '%'
				within_one_accuracy_perc_arr.append(within_one_accuracy_perc)
	# Calculate the percentage average and standard deviation of the accuracies
	within_one_accuracy_perc_avg = sum(within_one_accuracy_perc_arr) / float(len(within_one_accuracy_perc_arr))
	within_one_accuracy_stdev = statistics.stdev(within_one_accuracy_perc_arr)
	print 'Within-one-accuracy average and STD:'
	print within_one_accuracy_perc_avg, within_one_accuracy_stdev
