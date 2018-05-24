import sklearn, keras, csv, sys

ids_row_id = 0
ys_row_id = 1
simple_xs_row_ids = [1]
neural_xs_row_ids = [1,2,3]


def neuralRatingPredictor(ids, xs, ys):
	return NotImplementedError

def simpleRatingPredictor(ids, xs, ys):
	predictors = [sklearn.ensemble.RandomForestClassifier() for i in range(4)]

if __name__ == "__main__":
	with open(sys.argv[1], "rb") as in_f:
		reader = csv.reader(in_f)
		data = [row for row in reader]
		ids = [row[ids_row_id] for row in data]
		ys = [row[ys_row_id] for row in data]
		if sys.argv[2] == "simple":
			xs = [[row[id] for id in simple_xs_row_ids] for row in data]
			simpleRatingPredictor(ids,xs,ys)
		elif sys.argv[2] == "neural":
			x = [[row[id] for id in neural_xs_row_ids] for row in data]
