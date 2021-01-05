import numpy as np
import pdb
import pandas as pd

np.random.seed(0)

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

def traverse_tree(tree, mapback):
	# if tree.attr == 5:
	# 	pdb.set_trace()
	"""For debugging purposes"""
	if len(tree.branches) == 0:
		print("Terminate")

	print("Attribute: ", mapback[tree.attr])
	for val in tree.branches.keys():
		traverse_tree(tree.branches[val], mapback)

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
		val = plural_value(p_examples)
		return DecisionTree(lambda ex: val, -1, val)

	K = len(examples[0])-1
	labels = examples[:,K]
	if len(set(labels)) == 1:
		return DecisionTree(lambda ex: labels[0], -1, labels[0])
	elif len(attributes) == 0:
		val = plural_value(examples)
		return DecisionTree(lambda ex: val, -1, val)
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

def gen_categorical_data(N):
	patrons = ["None", "Some", "Full"]
	wait_time = ["10-30", "30-60", ">60"]
	hungry = ["No", "Yes"]
	alternate = ["No", "Yes"]
	reservation = ["No", "Yes"]
	weekend = ["No", "Yes"]
	bar = ["No", "Yes"]
	raining = ["No", "Yes"]
	should_wait = ["False", "True"]

	name_map = {
		"patrons": patrons,
		"wait_time": wait_time,
		"hungry": hungry,
		"alternate": alternate,
		"reservation": reservation,
		"weekend": weekend,
		"bar": bar,
		"raining": raining,
		"should_wait": should_wait
	}

	forseries = {}

	# learn how to use a pandas dataframe here
	for name in name_map.keys():
		arr = np.random.choice(name_map[name], N)
		forseries[name] = arr

	# generate the label
	arr = []
	for i in range(N):
		if forseries["hungry"][i] == "No" and forseries["alternate"][i] == "No" and forseries["raining"][i] == "No":
			arr.append("True")
		else:
			arr.append("False")
			#arr.append(np.random.choice(name_map["should_wait"]))
	forseries["should_wait"] = arr
	df = pd.DataFrame(forseries)
	#pdb.set_trace()

	# add a label - just do it randomly now
	return df

def construct_attributes_map(df):
	K = len(df.columns)
	attr_map = {}
	num_attr = 0
	for i in range(K):
		name = df.columns[i]
		vals_list = df[name].unique()
		if len(vals_list) > 2:
			num_attr += len(vals_list)
		else:
			num_attr += 1
		vals_map = {val : j for j, val in enumerate(vals_list)}
		attr_map[name] = vals_map
	return attr_map, num_attr

def binarize_matrix(df, attr_map, num_attr):
	N = len(df.index)
	X = np.zeros(shape=(N,num_attr), dtype=int)
	# remember to put label into the very last column
	cur = 0
	mapback = {}
	#pdb.set_trace()
	for name in attr_map.keys():
		print("Name: ", name)
		print("Cur: ", cur)
		print("Attribute map: ", attr_map[name])
		if len(attr_map[name]) > 2:
			# look at X
			print("padding")
			for i in range(N):
				val = df[name][i] 
				col = attr_map[name][val]
				X[i][cur+col] = 1
			mapback[cur] = name
			mapback[cur+1] = name
			mapback[cur+2] = name
			cur += 3
		else:
			#print(name)
			for i in range(N):
				val = df[name][i]
				X[i][cur] = attr_map[name][val] # does this work
			mapback[cur] = name
			cur += 1
	mapback[-1] = -1
	#pdb.set_trace()
	return X, mapback

def generate_plots():
	pass

def interpret_trees():
	pass

def run_experiment():
	df = gen_categorical_data(1000)
	attr_map, num_attr = construct_attributes_map(df)
	X, mapback = binarize_matrix(df, attr_map, num_attr)
	attributes = list(range(num_attr-1))
	tree = dt_learning(X[:800,:], attributes, X[:800,:])
	test_vals = X[:800,:]
	K = test_vals.shape[1]
	y_preds = []
	traverse_tree(tree, mapback)
	for i in range(len(test_vals)):
		y_preds.append(test(tree,test_vals[i,:K]))

	y_preds = np.array(y_preds)
	#pdb.set_trace()
	acc = np.mean(y_preds == test_vals[:,K-1])
	print(acc)

if __name__ == "__main__":
	#simple_test_3()
	#print("testing probabilistic...")
	#simple_test_4()
	run_experiment()
