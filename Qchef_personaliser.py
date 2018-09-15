# ToDo: Repeat experiments 3* for users 10* for recipes
# ToDo: Add the recipe specific features
# ToDo: Report the correlations

import os, sys, statistics, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
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
	# Evaluate the model
	model_accuracy = model.score(test_xset, test_raw_ys)
	print 'Using model.score', model_accuracy * 100, '%'
	# test_pred = model.predict(test_xset)
	test_pred = np.array(model.predict(test_xset))
	# print 'Predictions:', test_pred
	# test_pred = test_pred * 5
	test_pred_rounded = np.around(test_pred)
	# test_raw_ys = test_raw_ys * 5
	# print 'rounding the predictions:', len(test_pred_rounded), set(test_pred_rounded), test_pred_rounded
	model_accuracy, model_accuracy_norm = accuracy_score(test_raw_ys, test_pred_rounded, normalize=False), accuracy_score(test_raw_ys, test_pred_rounded)
	print 'Using rounded for accuracy_score:', model_accuracy, model_accuracy_norm * 100, '%'
	holdout_model_accuracy = mean_squared_error(test_raw_ys, test_pred)
	print 'Models mean_squared_error:', holdout_model_accuracy
	holdout_model_accuracy = mean_absolute_error(test_raw_ys, test_pred)
	print 'Models mean_absolute_error:', holdout_model_accuracy
	confusion_matrix_arr = confusion_matrix(test_raw_ys, test_pred_rounded, labels=list(set(test_raw_ys)))
	print 'Confusion matrix:\n', confusion_matrix_arr
	within_one_accuracy = within_one_accuracy_fn(confusion_matrix_arr)
	within_one_accuracy_percentage = within_one_accuracy / float(len(test_raw_ys)) * 100
	print 'within_one_accuracy', within_one_accuracy, within_one_accuracy_percentage, '%'
	# model_recall_fscore = precision_recall_fscore_support(test_raw_ys, test_pred, average='macro')
	# print 'Using precision_recall_fscore_support:', model_recall_fscore
	model_recall_fscore = precision_recall_fscore_support(test_raw_ys, test_pred_rounded, average='macro')
	print 'Using rounded for precision_recall_fscore_support:', model_recall_fscore
	spearmanr_corr = spearmanr(test_target_var, test_pred)
	pearsonr_corr = pearsonr(test_target_var, test_pred)
	return within_one_accuracy_percentage, spearmanr_corr, pearsonr_corr

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
	holdout_model_accuracy = mean_squared_error(test_yset, test_pred)
	print 'Models mean_squared_error:', holdout_model_accuracy
	holdout_model_accuracy = mean_absolute_error(test_yset, test_pred)
	print 'Models mean_absolute_error:', holdout_model_accuracy
	keras_holdout_model_accuracy = keras_mean_squared_error(test_yset, test_pred)
	print 'Models keras_holdout_model_accuracy:', keras_holdout_model_accuracy
	holdout_model_recall_fscore = precision_recall_fscore_support(test_yset, test_pred_rounded, average='macro')
	print 'Using precision_recall_fscore_support:', holdout_model_recall_fscore
	confusion_matrix_arr = confusion_matrix(test_yset, test_pred_rounded, labels=list(set(test_yset)))
	print 'Confusion matrix:\n', confusion_matrix_arr
	within_one_accuracy = within_one_accuracy_fn(confusion_matrix_arr)
	within_one_accuracy_percentage = within_one_accuracy / float(len(test_yset)) * 100
	print 'within_one_accuracy', within_one_accuracy, within_one_accuracy_percentage, '%'
	model_recall_fscore = precision_recall_fscore_support(test_yset, test_pred_rounded, average='macro')
	print 'Using precision_recall_fscore_support:', model_recall_fscore
	spearmanr_corr = spearmanr(test_target_var, test_pred.T[0])
	pearsonr_corr = pearsonr(test_target_var, test_pred.T[0])
	return within_one_accuracy_percentage, spearmanr_corr, pearsonr_corr

def within_one_accuracy_fn(__confusion_matrix_arr__):
	__within_one_accuracy__ = 0
	for each_label in range(len(__confusion_matrix_arr__) - 1):
		__within_one_accuracy__ += __confusion_matrix_arr__[each_label][each_label] + \
								   __confusion_matrix_arr__[each_label][each_label + 1] + \
								   __confusion_matrix_arr__[each_label + 1][each_label]
	__within_one_accuracy__ += __confusion_matrix_arr__[-1][-1]
	return __within_one_accuracy__

if __name__ == '__main__':
	# Get the argument variables
	user_input_fn = sys.argv[1]
	algo_mode = sys.argv[2]
	food_cuisine_survey_fp = sys.argv[3]
	split_mode = sys.argv[4]
	print 'Algorithm:', algo_mode
	print 'Split mode:', split_mode
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
	# Read the prepared user input
	temp_user_input_df = pd.read_csv(user_input_fn)
	print 'Number of records:', len(temp_user_input_df)
	print 'Unique surprise ratings:', temp_user_input_df['users_surp_ratings'].unique()
	print 'Number of unique surprise ratings:', temp_user_input_df['Recipe ID'].nunique()
	# Remove unsure records
	temp_user_input_df = temp_user_input_df[temp_user_input_df['users_surp_ratings'] != -0.2]
	print 'After filtering out the unsure records:'
	print 'Number of records:', len(temp_user_input_df)
	print 'Unique surprise ratings:', temp_user_input_df['users_surp_ratings'].unique()
	print 'Number of unique surprise ratings:', temp_user_input_df['Recipe ID'].nunique()
	# Initialize ID list, KFold column name and DataFrames
	id_list, kf, col_name, train_df, test_df = [], KFold(), '', pd.DataFrame(), pd.DataFrame()
	# Initialize dict of predictors to store the predictors of each fold
	predictors_dict = {}
	within_one_accuracy_arr = []
	spearmanr_arr = []
	pearsonr_arr = []
	# Split the training-validation by: users or recipes
	if split_mode == 'user':
		# Determine the column to use for splitting: user IDs
		col_name = 'User ID'
		# Get the list o IDs
		id_list = temp_user_input_df[col_name].unique()
		# Split on 10% of the users to the average of 10 models
		kf = KFold(n_splits=10, shuffle=True)
	elif split_mode == 'recipe':
		# Determine the column to use for splitting: recipe IDs
		col_name = 'Recipe ID'
		# Get the list o IDs
		id_list = temp_user_input_df[col_name].unique()
		# Split on 1 recipe to the test to get the average 16 models
		kf = KFold(n_splits=16, shuffle=True)
	# Iterate over the splits
	for fold_idx, (train_ids, test_ids) in enumerate(kf.split(id_list)):
		# train_ids are indecies of the IDs not the IDs themselves, so id_list[train_ids] for getting those IDs
		temp_train_df = temp_user_input_df[temp_user_input_df[col_name].isin(id_list[train_ids])]
		temp_test_df = temp_user_input_df[temp_user_input_df[col_name].isin(id_list[test_ids])]
		# Remove user and recipe IDs and make a new separate DF without those IDs, because we want to use the IDs in the next iterations
		user_input_df = temp_user_input_df.drop(['Recipe ID', 'User ID'], axis=1)
		train_df = temp_train_df.drop(['Recipe ID', 'User ID'], axis=1)
		test_df = temp_test_df.drop(['Recipe ID', 'User ID'], axis=1)
		# Remove unwanted features; choose to drop any of the following:
		# Lists of column names: users_fam_cols, users_cuisine_pref_cols, knowledge_ingredient_cols or cuisine_knowledge_cols, users_surp_pref_cols, cuisine_softmax_cols
		# Individual column names:
		# 	'observed_surp_estimates_90perc', 'observed_surp_estimates_95perc', 'observed_surp_estimates_max',
		# 	'oracle_surp_estimates_90perc', 'oracle_surp_estimates_95perc', 'oracle_surp_estimates_max',
		# 	'personalized_surp_estimates_90perc', 'personalized_surp_estimates_95perc', 'personalized_surp_estimates_max',
		# 	'users_surp_pref'
		dropped_cols = []
		user_input_df.drop(dropped_cols, axis=1, inplace=True)
		train_df.drop(dropped_cols, axis=1, inplace=True)
		test_df.drop(dropped_cols, axis=1, inplace=True)
		# print 'Current used features:', user_input_df.columns
		# Get the target variable
		target_var = user_input_df['users_surp_ratings']
		train_target_var = train_df['users_surp_ratings']
		test_target_var = test_df['users_surp_ratings']
		# Scale the target values to ratings 1-5
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
		# Get number of used predictive variables
		num_predictive_vars = np.shape(predictive_var_arr)[1]
		print 'Number of predictive variables:', num_predictive_vars
		# Choose and build the model
		if algo_mode == 'RF_regressor':
			within_one_accuracy, spearmanr_corr, pearsonr_corr = \
				RF_regressor(train_predictive_var_arr, train_target_var, test_predictive_var_arr, test_target_var)
			within_one_accuracy_arr.append(within_one_accuracy)
			spearmanr_arr.append(spearmanr_corr)
			pearsonr_arr.append(pearsonr_corr)
		elif algo_mode == 'neural':
			within_one_accuracy, spearmanr_corr, pearsonr_corr = \
				neuralRatingPredictor(train_predictive_var_arr, train_target_var, test_predictive_var_arr, test_target_var)
			within_one_accuracy_arr.append(within_one_accuracy)
			spearmanr_arr.append(spearmanr_corr)
			pearsonr_arr.append(pearsonr_corr)
		else:
			print 'Sorry! No algorithm chosen!'
			sys.exit()
	avg_within_one_accuracy = reduce(lambda x, y: x + y, within_one_accuracy_arr) / len(within_one_accuracy_arr)
	std_within_one_accuracy = statistics.stdev(within_one_accuracy_arr)
	print 'Average of within one accuracy:', avg_within_one_accuracy, '%,', 'STD:', std_within_one_accuracy, '%'
	# Calculate the average pearson and spearman correlations
	# print 'spearmanr_arr', spearmanr_arr
	print 'average spearmanr correlation', sum([each_corr[0] for each_corr in spearmanr_arr]) / float(len(spearmanr_arr))
	print 'std spearmanr correlation', statistics.stdev([each_corr[0] for each_corr in spearmanr_arr])
	print 'average spearmanr p-value', sum([each_corr[1] for each_corr in spearmanr_arr]) / float(len(spearmanr_arr))
	print 'std spearmanr p-value', statistics.stdev([each_corr[1] for each_corr in spearmanr_arr])
	# print 'pearsonr_arr', pearsonr_arr
	print 'average pearsonr correlation', sum([each_corr[0] for each_corr in pearsonr_arr]) / float(len(pearsonr_arr))
	print 'std pearsonr correlation', statistics.stdev([each_corr[0] for each_corr in pearsonr_arr])
	print 'average pearsonr p-value', sum([each_corr[1] for each_corr in pearsonr_arr]) / float(len(pearsonr_arr))
	print 'std pearsonr p-value', statistics.stdev([each_corr[1] for each_corr in pearsonr_arr])
