import numpy as np
import pdb


"""Implementing a boolean decision tree for pedagogical purposes"""

class DecisionTree:
	"""Decision Tree class"""
	def __init__(self, attr_test, attr, val):
		self.test = attr_test
		self.attr = attr
		self.weights = {}
		self.branches = {}
		self.val = val

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
	Xs = np.random.randint(low=0, high=2, size=(n,K+1), dtype=int)
	# let's make this a threshold function
	w = np.random.randint(low=0, high=2, size=(K,), dtype=int)
	Ys = [1 if np.dot(Xs[i][:K], w) > 1 else 0 for i in range(n)] # does this work? lol FIX this indexing!!
	for i in range(len(Ys)):
		Xs[i][K] = Ys[i]
	return Xs

# inputs are of form X \in [0,1]^K
# outputs are of form Y \in [0,1]

"""
examples should be a 2d matrix
attributes is the set of indices into X in examples
"""
def dt_learning(examples, attributes, p_examples):
	if len(examples) == 0:
		return plural_value(p_examples)

	K = len(examples[0])-1
	labels = examples[:,K]
	if len(set(labels)) == 1:
		return DecisionTree(lambda ex: labels[0], -1, labels[0])
	elif len(attributes) == 0:
		return plural_value(examples)
	else:
		importances = [(a,importance(examples, a)) for a in attributes]
		importances = sorted(importances,key=lambda x:(-x[1],x[0]))
		attributes = [imp[0] for imp in importances]
		attr = importances[0][0]
		tree = create_tree(attr, examples)
		attributes.pop(0)
		for k in range(2): 
			idxs = np.argwhere(examples[:,attr] == k)
			tree.weights[k] = float(len(idxs)) / len(examples)
			idxs = [idx[0] for idx in idxs]
			exs = examples[idxs]
			tree.branches[k] = dt_learning(exs, attributes, examples)
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
	#pdb.set_trace()

def simple_test_3():
	examples = [[1,1,0,1],
				[1,0,0,0],
				[0,1,0,0],
				[0,1,1,0],
				[1,1,1,1],
				[0,0,1,0],
				[0,0,0,0],
				[1,0,1,0]]
	examples = np.array(examples)
	attributes = list(range(3))
	tree = dt_learning(examples, attributes, examples)
	print("traversing tree")
	traverse_tree(tree)
	result = test(tree, [1,1,0])
	print(result)
	result = test(tree, [1,0,0])
	print(result)
	result = test(tree, [0,1,0])
	print(result)
	result = test(tree, [0,1,1])
	print(result)
	result = test(tree, [1,1,1])
	print(result)
	result = test(tree, [0,0,1])
	print(result)
	result = test(tree, [0,0,0])
	print(result)
	result = test(tree, [1,0,1])
	print(result)

def simple_test_4():
	examples = [[1,1,0,1],
				[1,0,0,0],
				[0,1,0,0],
				[0,1,1,0],
				[1,1,1,1],
				[0,0,1,0],
				[0,0,0,0],
				[1,0,1,0]]
	examples = np.array(examples)
	attributes = list(range(3))
	tree = dt_learning(examples, attributes, examples)
	print("traversing tree")
	traverse_tree(tree)
	result = test_probabilistic(tree, [1,1,0])
	print(result)
	result = test_probabilistic(tree, [1,0,0])
	print(result)
	result = test_probabilistic(tree, [0,1,0])
	print(result)
	result = test_probabilistic(tree, [0,1,1])
	print(result)
	result = test_probabilistic(tree, [1,1,1])
	print(result)
	result = test_probabilistic(tree, [0,0,1])
	print(result)
	result = test_probabilistic(tree, [0,0,0])
	print(result)
	result = test_probabilistic(tree, [1,0,1])
	print(result)

def simple_test_2():
	K = 10
	Xs= gen_data(100,K)
	print(Xs[0])
	attributes = list(range(K))
	tree = dt_learning(Xs, attributes, Xs) 
	result = test(tree, Xs[0][:K])
	print(result)

if __name__ == "__main__":
	simple_test_3()
	print("testing probabilistic...")
	simple_test_4()
