import tensorflow as tf
import h5py

def loadH5(hdf5_file, var_list=None, verbose=True):
	import h5py, numpy as np
	if var_list == None:
		var_list = tf.trainable_variables()
	h = h5py.File(hdf5_file, 'r')
	d = h['data']
	weights_and_biases = []
	for k in d:
		if list(d[k]):
			weight = np.array(d[k]['0'])
			bias = None
			if len(d[k]) > 1:
				bias = np.array(d[k]['1'])
			if len(weight.shape) == 4: # Convolution
				weight = weight.transpose([2,3,1,0])
			weights_and_biases.append((k,weight,bias))
	tf_biases = {}
	for v in var_list:
		if '/biases' in v.op.name:
			tf_biases[v.op.name.replace('/biases','')] = v
	ops = []
	used = [False] * len(weights_and_biases)
	for v in var_list:
		if '/weights' in v.op.name:
			init = False
			for i in range(len(weights_and_biases)):
				if not used[i] and weights_and_biases[i][1].shape == v.get_shape():
					weight, bias = weights_and_biases[i][1:]
					used[i] = True
					basename = v.op.name.replace('/weights','')
					vb = None
					if basename in tf_biases and bias is not None:
						# Copy the bias
						vb = tf_biases[basename]
						assert vb.get_shape() == bias.shape, "Bias shape does not match"
						ops.append(vb.assign(bias))
					ops.append(v.assign(weight))
					if verbose:
						print( '%-30s matches %30s'%(v.op.name, weights_and_biases[i][0]))
					init = True
					break
			if not init and verbose:
				print( "%-30s no match found, not loaded!"%v.op.name)
	h.close()
	return tf.group(*ops)
