from collections import Counter
from helpers import dot_product, multiply


def train(x, y, l):
	j = 0
	max_iter = 20
	w = Counter({w: 0 for w in x[0]})

	print "Lambda = ",l

	while j < max_iter:

		t = 0.0

		print "Passing through data, iter #", j + 1
		for i, features in enumerate(x):

			t += 1.0
			nt = 1/(t*l)

			p1 = 1 - nt * l
			w  = multiply(p1, w)

			op = multiply(y[i], w)

			if dot_product(op, features) < 1:
				p2 = multiply(nt*y[i], features)
				w.update(p2)

		j += 1

	return w
