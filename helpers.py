from collections import Counter


def read_instances(name):
	f = open(name, "r")
	lines = f.readlines()
	return lines


def build_vocabulary(lines):
	vc = Counter()

	for line in lines:
		line_tmp = parse_line(line)

		line_occr = Counter()
		for i, word in enumerate(line_tmp):
			line_occr[word] = 1

		vc += line_occr

	return vc


def parse_line(line):
	tmp = line.split(" ")[1:]
	tmp[-1] = tmp[-1].rstrip("\n")
	return tmp


def build_feature(line, vocabulary):
	feature_vector = Counter()

	line_tmp = parse_line(line)

	for word in vocabulary.keys():
		feature_vector[word] = 1 if word in line_tmp else 0

	return feature_vector


def build_features(lines, vocabulary):
	features_list = list()

	for line in lines:
		feature_vector = build_feature(line, vocabulary)
		features_list.append(feature_vector)

	return features_list


def build_labels(lines):
	desired_outputs = list()

	for line in lines:
		y = -1 if line.split(" ")[0] == '0' else 1
		desired_outputs.append(y)

	return desired_outputs


def dot_product(a, b):
	return sum((Counter({w: a[w] * b[w] for w in a})).values())


def multiply(number, counter):
	return Counter({word: (v * number) for (word, v) in counter.iteritems()})


def test(w, x, y):
	hits = 0

	for i, features in enumerate(x):
		result = dot_product(features, w)
		hits += 1 if result >= 0 and y[i] == 1 or result < 0 and y[i] == -1 else 0

	return 1 - hits/float(len(y))