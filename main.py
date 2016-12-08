from helpers import *
import perceptron
import pegasos

# Limit perceptron training iterations
PERCEPTRON_MAX_ITER = 30

print "Reading training file..."
lines = read_instances("spam_train.txt")

print "Counting words..."
vocabulary = build_vocabulary(lines)

print "Building labels..."
desired_outputs = build_labels(lines)

print "Filtering instances..."
vocabulary = Counter({w: c for w, c in vocabulary.iteritems() if c >= 30})

print "Buiding feature vectors..."
feature_vectors = build_features(lines, vocabulary)

print "Calling perceptron train"
(w, k, it) = perceptron.train(feature_vectors, desired_outputs, PERCEPTRON_MAX_ITER)

print "Number of iterations (perceptron):", it

print "Number of mistakes (perceptron):", k

print "15 most positive words (perceptron):", w.most_common(15)

print "15 most negative words: (perceptron)", w.most_common()[:-15-1:-1]

print "Calling pegasos train"

# Test with array of lambdas or hardcoded one

l = range(-9,9)
ws = [pegasos.train(feature_vectors, desired_outputs, 2**x) for x in l]

#ws = [pegasos.train(feature_vectors, desired_outputs, 2 ** -7)]

print "Reading validation file..."
lines = read_instances("spam_val.txt")

print "Building feature vectors (validation)..."
feature_vectors = build_features(lines, vocabulary)

print "Building labels (validation)..."
desired_outputs = build_labels(lines)

print "Calling perceptron test..."
error = test(w, feature_vectors, desired_outputs)

print "Perceptron error on validation:", error, "%"

for w in ws:
	print "Calling pegasos test..."
	error = test(w, feature_vectors, desired_outputs)
	print "Pegasos Error on validation:", error, "%"
