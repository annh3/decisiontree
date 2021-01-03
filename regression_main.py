import numpy as np
import pdb


"""Implementing a boolean decision tree for pedagogical purposes"""

class RegressionDecisionTree:
	"""Decision Tree class"""
	def __init__(self, attr_test, attr, val):
		self.test = attr_test
		self.attr = attr
		self.weights = {}
		self.branches = {}
		self.val = val # this is the learned value

def test(tree, x):
	# base case
	if len(tree.branches) == 0:
		return tree.val
	output = tree.test(x)
	return test(tree.branches[output], x)

def test_probabilistic(tree, x):
	if len(tree.branches) == 0:
		return tree.val

	if x[tree.attr] == -1:
		val1 = test_probabilistic(tree.branches[0], x)
		val2 = test_probabilistic(tree.branches[1], x)
		output = (tree.weights[0]*val1 + tree.weights[1]*val2 > 0.5).astype(int)
	else:
		output = tree.test(x)

	return test_probabilistic(tree.branches[output], x)

def traverse_tree(tree):
	"""For debugging purposes"""
	if len(tree.branches) == 0:
		print("Terminate")

	print("Attribute: ", tree.attr)
	for val in tree.branches.keys():
		traverse_tree(tree.branches[val])

def plural_value(p_examples):
	K = len(p_examples[0])-1
	p_labels = p_examples[:,K]
	counts = np.bincount(p_labels)
	return counts[np.argmax(counts)] # does this work

def calculate_entropy(p):
	if abs(p - 0) < 0.000001: return 0
	if abs(p - 1) < 0.000001: return 0
	return -(p* np.log2(p) + (1-p) * np.log2(1-p))

def importance(examples, attribute):
	"""Calculate the information gain as measured by entropy"""

	# Gain = (p / (p + n)) - Remainder()
	# p proportion 1's on [attribute]
	K = len(examples[0])-1
	labels = examples[:,K]
	n = np.count_nonzero(labels == 0)
	p = np.count_nonzero(labels == 1) # no, we need to

	idxs_b = np.argwhere(examples[:,attribute] == 1) 
	idxs_b = [idx[0] for idx in idxs_b]

	idxs_c = np.argwhere(examples[:,attribute] == 0)
	idxs_c = [idx[0] for idx in idxs_c]
	b = examples[idxs_b]
	c = examples[idxs_c]

	n_b = np.count_nonzero(b[:,K] == 0)
	p_b = np.count_nonzero(b[:,K] == 1)

	n_c = np.count_nonzero(c[:,K] == 0)
	p_c = np.count_nonzero(c[:,K] == 1)

	val1 = calculate_entropy(p/float(p+n))
	val2 = 0
	if len(b) > 0:
		val2 = calculate_entropy(p_b/float(p_b+n_b))
	val3 = 0
	if len(c) > 0:
		val3 = calculate_entropy(p_c/float(p_c+n_c))

	weight = float(len(b))/(len(b)+len(c))

	return val1 - (weight*val2 + (1-weight)*val3)

def calculate_weights(examples, attribute):
	K = len(examples[0])-1
	labels = examples[:,K]
	n = len(labels)
	p = np.count_nonzero(labels == 1)

	return 1 - p / float(p+n), p / float(p+n)

def test_value(examples, attribute):
	# To-Do: clean this up
	K = len(examples[0])-1
	labels = examples[:,K]
	n = np.count_nonzero(labels == 0)
	p = np.count_nonzero(labels == 1) 

	idxs_b = np.argwhere(examples[:,attribute] == 1) 
	idxs_b = [idx[0] for idx in idxs_b]

	idxs_c = np.argwhere(examples[:,attribute] == 0)
	idxs_c = [idx[0] for idx in idxs_c]
	b = examples[idxs_b]
	c = examples[idxs_c]

	n_b = np.count_nonzero(b[:,K] == 0)
	p_b = np.count_nonzero(b[:,K] == 1)

	n_c = np.count_nonzero(c[:,K] == 0)
	p_c = np.count_nonzero(c[:,K] == 1)

	if p_b > p_c:
		return 1
	else:
		return 0


def create_tree(attribute, examples):
	val = test_value(examples, attribute)
	return DecisionTree(lambda ex: int(ex[attribute] == 1), attribute, val) # lol will this compile?

def gen_data(n,K):
	# generate attribute 1
	# generate attribute 2
	# generate attribute 3
	# generate attribute 4
	# generate labels according to polynomial function
	# vstack the columns together

	attr1 = np.random.uniform(low=20.0, high=70.0, size=n)
	attr2 = np.random.uniform(low=1000.0, high=5000.0, size=n)
	attr3 = np.random.uniform(low=0.0, high=20.0, size=n)
	attr4 = np.random.uniform(low=10.0, high=15.0, size=n)

	labels = np.power((np.power(attr1,3) - attr2) + np.multiply(attr3, attr4), 0.1)
	Xs = np.column_stack((attr1, attr2, attr3, attr4, labels))
	#pdb.set_trace()
	return Xs

def calculate_loss(examples, thresh):
	#print("threshold: ", thresh)
	K = len(examples[0])-1
	class0_idxs = np.argwhere(examples[:,K] < thresh)
	class0_idxs = [idx[0] for idx in class0_idxs]
	ex_class0 = examples[class0_idxs]
	label0 = np.mean(ex_class0[:,K])

	class1_idxs = np.argwhere(examples[:,K] >= thresh)
	class1_idxs = [idx[0] for idx in class1_idxs]
	ex_class1 = examples[class1_idxs]
	label1 = np.mean(ex_class1[:,K])

	n = len(examples)

	if len(ex_class0) == 0:
		array1 = ex_class1[:,K] - label1
		loss = np.sum(array1**2)
		#print("loss: ", loss)
		#pdb.set_trace()
		return loss, 0, label1
	if len(ex_class1) == 0:
		array0 = ex_class0[:,K] - label0
		loss = np.sum(array0**2)
		#print("loss: ", loss)
		#pdb.set_trace()
		return loss, label0, 0

	array0 = ex_class0[:,K] - label0
	array1 = ex_class1[:,K] - label1
	loss = np.sum(array0**2) + np.sum(array1**2)
	#print("loss: ", loss)
	#assert(1==0)
	#pdb.set_trace()
	return loss, label0, label1

def search_possible_splits(examples, attribute):
	# sort examples[:, attribute] 
	# how do we do this search?
	# first pass: naively split between each ex_i, ex_{i+1}
	vals = sorted(examples[:,attribute])
	losses = []
	for i in range(len(vals)-1):
		mid = (vals[i] + vals[i+1]) / 2
		loss, mean0, mean1 = calculate_loss(examples, mid)
		losses.append((loss, mid, mean0, mean1))
	#pdb.set_trace()
	#print(losses)
	#print("\n")
	losses = sorted(losses, key=lambda x:(-x[0], x[1], x[2], x[3]))
	if len(losses) == 0:
		pdb.set_trace()
		return (-1,-1,-1,-1)
	# return the best split
	#pdb.set_trace()
	return losses[0]

# inputs are of form X \in [0,1]^K
# outputs are of form Y \in [0,1]

"""
examples should be a 2d matrix
attributes is the set of indices into X in examples
"""

 # def __init__(self, attr_test, attr, val):
def dt_learning(examples, attributes, p_examples, delta):
	if len(examples) == 0:
		K = len(p_examples[0])-1
		return RegressionDecisionTree(lambda ex: -1, -1, np.mean(p_examples[:,K]))

	if len(attributes) == 0:
		K = len(examples[0])-1
		return RegressionDecisionTree(lambda ex: -1, -1, np.mean(examples[:,K]))

	if len(examples) == 1:
		K = len(examples[0])-1
		return RegressionDecisionTree(lambda ex: -1, -1, examples[0][K])

	K = len(examples[0])-1
	labels = examples[:,K]

	# take the mean value
	current_mean = np.mean(examples[:,K])
	S_orig = np.sum((examples[:,K]-current_mean)**2) # does this work?
	
	# find the best split
	vals = []
	for attr in attributes:
		loss, thresh, mean0, mean1 = search_possible_splits(examples, attr)
		print(loss)
		# calculate hte delta?
		vals.append((S_orig-loss, attr, thresh))
	vals = sorted(vals, key=lambda x:(-x[0], x[1], x[2]))
	#pdb.set_trace()
	# if len(vals) == 0:
	# 	pdb.set_trace()
	if vals[0][0] < delta:
		# don't split
		# return decision tree, but what are the details here?
		# we can just return mean for now
		# but this is where we can learn a regression model
		return RegressionDecisionTree(lambda ex: labels[0], -1, current_mean)
	else:
		# split on the best attribute and threshold
		tree = RegressionDecisionTree(lambda ex: ex[vals[0][1]] < vals[0][2], vals[0][1], current_mean)
		idxs = np.argwhere(examples[:,vals[0][1]] < vals[0][2])
		tree.weights[0] = float(len(idxs)) / len(examples)
		idxs = [idx[0] for idx in idxs]
		exs = examples[idxs]

		attributes = [val[1] for val in vals]
		attributes.pop(0)
		tree.branches[0] = dt_learning(exs, attributes, examples, delta)

		idxs = np.argwhere(examples[:,vals[0][1]] >= vals[0][2])
		tree.weights[1] = float(len(idxs)) / len(examples)
		idxs = [idx[0] for idx in idxs]
		exs = examples[idxs]
		tree.branches[1] = dt_learning(exs, attributes, examples, delta)

	return tree

def simple_test():
	examples = [[1,0,0,1], 
				[1,0,1,1],
				[0,0,1,0],
				[0,1,0,0],
				[1,1,1,1],
				[0,0,0,0]]
	examples = np.array(examples)
	attributes = list(range(3))
	tree = dt_learning(examples, attributes, examples)
	result = test(tree, [1,0,0])
	print(result)
	result = test(tree, [1,0,1])
	print(result)
	result = test(tree, [0,0,1])
	print(result)
	result = test(tree, [0,1,0])
	print(result)
	result = test(tree, [1,1,0])
	print(result)

if __name__ == "__main__":
	Xs = gen_data(100,4)
	attributes = list(range(Xs.shape[1]-1))
	tree = dt_learning(Xs, attributes, Xs, 0.0000000)
	traverse_tree(tree)
