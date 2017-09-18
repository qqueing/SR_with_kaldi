import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, trainable=True):

	with tf.device('/cpu:0'):
		dtype = tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
	return var

def BatchNorm(inputs, is_training, decay = 0.9, epsilon=1e-3, scale = True):

	scale = _variable_on_cpu('scale', inputs.get_shape()[-1], tf.constant_initializer(1.0))
	beta = _variable_on_cpu('beta', inputs.get_shape()[-1], tf.constant_initializer(0.0))
	pop_mean = _variable_on_cpu('mean', inputs.get_shape()[-1], tf.constant_initializer(0.0), trainable=False)
	pop_var = _variable_on_cpu('variance', inputs.get_shape()[-1], tf.constant_initializer(1.0), trainable=False)
	axis = list(range(len(inputs.get_shape())-1))

	def Train(inputs, pop_mean, pop_var, scale, beta):
		batch_mean, batch_var = tf.nn.moments(inputs,axis)
		train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean,train_var]):
			return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

	def Eval(inputs, pop_mean, pop_var, scale, beta):
		return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

	return tf.cond(is_training, lambda: Train(inputs, pop_mean, pop_var, scale, beta),
		lambda: Eval(inputs, pop_mean, pop_var, scale, beta))


