import argparse
import numpy as np 
import os
from typing import List, Optional, Any


# Import TensorFlow
import tensorflow as tf

from quad_gen.code_blocks import (
	headers_network_evaluate,
	linear_activation,
	sigmoid_activation,
	relu_activation,
)


def generate(policy: Any, sess: Any, output_path: Optional[str] = None) -> str:
	"""
	Generate mlp model source code given a policy object
	Args:
		policy [policy object]: the trained policy (can be JAX/Flax params or TensorFlow)
		sess [tf.compat.v1.Session]: a tensorflow session
		output_path [str, optional]: the path of the generated code (should include the file name)
	Returns:
		str: the generated source code
	"""
	# Handle JAX/Flax parameter structure
	if hasattr(policy, 'get') and callable(policy.get):
		# This is likely a JAX/Flax FrozenDict or similar
		trainable_list = []
		trainable_shapes = []
		trainable_evals = []
		
		# Extract parameters from JAX/Flax structure
		# The policy has a 'params' key containing the actual parameters
		if 'params' in policy:
			params = policy['params']
			for layer_name in ['hidden_0', 'hidden_1', 'hidden_2', 'hidden_3', 'hidden_4']:
				if layer_name in params:
					layer_params = params[layer_name]
					# Add weights first, then biases
					if 'kernel' in layer_params:
						trainable_list.append(layer_params['kernel'])
						trainable_shapes.append(layer_params['kernel'].shape)
						trainable_evals.append(layer_params['kernel'])
					if 'bias' in layer_params:
						trainable_list.append(layer_params['bias'])
						trainable_shapes.append(layer_params['bias'].shape)
						trainable_evals.append(layer_params['bias'])
	else:
		# Original TensorFlow handling
		trainable_list = policy.get_params()
		trainable_shapes = []
		trainable_evals = []
		for tf_trainable in trainable_list:
			trainable_shapes.append(tf_trainable.shape)
			trainable_evals.append(tf_trainable.eval(session=sess))

	"""
	To account for the last matrix which stores the std, 
	the # of layers must be subtracted by 1
	"""
	n_layers = len(trainable_shapes) - 1
	weights: List[str] = []	# strings
	biases: List[str] = []		# strings
	outputs: List[str] = []	# strings

	structure = f"""static const int structure[{int(n_layers/2 + 1)}][2] = {{""" # structure code thingy

	n_weight = 0
	n_bias = 0
	for n in range(n_layers): 
		shape = trainable_shapes[n]
		
		if len(shape) == 2:
			# it is a weight matrix
			weight = f"""static const float layer_{n_weight}_weight[{shape[0]}][{shape[1]}] = {{"""
			for row in trainable_evals[n]:
				weight += """{"""
				for num in row:
					weight += f"{num},"
				# get rid of the comma after the last number
				weight = weight[:-1]
				weight += """},"""
			# get rid of the comma after the last curly bracket
			weight = weight[:-1]
			weight += """};\n"""
			weights.append(weight)
			n_weight += 1

			# augment the structure array
			structure += f"""{{{shape[0]}, {shape[1]}}},"""

		elif len(shape) == 1:
			# it is a bias vector 
			bias = f"""static const float layer_{n_bias}_bias[{shape[0]}] = {{"""
			for num in trainable_evals[n]:
				bias += f"{num},"
			# get rid of the comma after the last number
			bias = bias[:-1]
			bias += """};\n"""
			biases.append(bias)

			# add the output arrays
			output = f"""static float output_{n_bias}[{shape[0]}];\n"""
			outputs.append(output)

			n_bias += 1

	# complete the structure array
	# get rid of the comma after the last curly bracket
	structure = structure[:-1] 
	structure += """};\n"""

	"""
	Multiple for loops to do matrix multiplication
	 - assuming using tanh activation
	"""
	for_loops: List[str] = []	# strings 

	# the first hidden layer
	input_for_loop = """
		for (int i = 0; i < structure[0][1]; i++) {
			output_0[i] = 0;
			for (int j = 0; j < structure[0][0]; j++) {
				output_0[i] += state_array[j] * layer_0_weight[j][i];
			}
			output_0[i] += layer_0_bias[i];
			output_0[i] = tanhf(output_0[i]);
		}
	"""
	for_loops.append(input_for_loop)

	# rest of the hidden layers
	for n in range(1, int(n_layers/2)-1):
		for_loop = f"""
		for (int i = 0; i < structure[{n}][1]; i++) {{
			output_{n}[i] = 0;
			for (int j = 0; j < structure[{n}][0]; j++) {{
				output_{n}[i] += output_{n-1}[j] * layer_{n}_weight[j][i];
			}}
			output_{n}[i] += layer_{n}_bias[i];
			output_{n}[i] = tanhf(output_{n}[i]);
		}}
		"""
		for_loops.append(for_loop)

	n = int(n_layers/2)-1
	# the last hidden layer which is supposed to have no non-linearity
	output_for_loop = f"""
		for (int i = 0; i < structure[{n}][1]; i++) {{
			output_{n}[i] = 0;
			for (int j = 0; j < structure[{n}][0]; j++) {{
				output_{n}[i] += output_{n-1}[j] * layer_{n}_weight[j][i];
			}}
			output_{n}[i] += layer_{n}_bias[i];
		}}
		"""
	for_loops.append(output_for_loop)

	# assign network outputs to control
	assignment = f"""
		control_n[0] = output_{n}[0];
		control_n[1] = output_{n}[1];
		control_n[2] = output_{n}[2];
		control_n[3] = output_{n}[3];	
	"""






	# construct the network evaluation function
	controller_eval = """
	void networkEvaluate(float *state_array, float *control_n) {
	"""
	for code in for_loops:
		controller_eval += code 
	# assignment to control_n
	controller_eval += assignment

	# closing bracket
	controller_eval += """
	}
	"""

	# combine the all the codes
	source = ""
	# headers
	source += headers_network_evaluate
	# helper functions
	source += linear_activation
	source += sigmoid_activation
	source += relu_activation
	# the network evaluation function
	source += structure
	for output in outputs:
		source += output 
	for weight in weights:
		source += weight 
	for bias in biases:
		source += bias
	source += controller_eval

	# add log group for logging
	# source += log_group

	if output_path:
		with open(output_path, 'w') as f:
			f.write(source)

	return source