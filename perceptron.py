from collections import Counter
from helpers import dot_product, multiply


def train(x, y, max_iter):
	mistakes = iterations = 0
	w = Counter({w: 0 for w in x[0]})

	while True:

		all_correct = True
		iterations += 1

		print "Passing through data, iter #", iterations
		for i, features in enumerate(x):

			op = multiply(y[i], features)

			zeros = all(map(lambda v: v == 0, features.values()))
			if zeros: continue

			if dot_product(op, w) <= 0:
				w.update(op)
				all_correct = False
				mistakes += 1

		if all_correct or max_iter == iterations:
			break

	return w, mistakes, iterations
