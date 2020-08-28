from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GGCL_F(Layer):
    """GGCL: the input is feature"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, bias=False,
                 featureless=False, **kwargs):
        super(GGCL_F, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_0'] = glorot([input_dim, output_dim], name='weights_0')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        supports = list()
        i = 0
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_' + str(i)],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_' + str(i)]
        support = dot(self.support[i], pre_sup, sparse=True)
        supports.append(support)
        dim = int(self.output_dim / 2)
        mean_vector = tf.nn.elu(tf.slice(pre_sup, [0, 0], [-1, dim]))
        var_vector = tf.nn.relu(tf.slice(pre_sup, [0, dim], [-1, dim]))
        self.vars['mean'] = mean_vector
        self.vars['var'] = var_vector
        node_weight = tf.exp(-var_vector*FLAGS.para_var)
        mean_out = dot(self.support[0], mean_vector * node_weight, sparse=True)
        var_out = dot(self.support[1], var_vector * node_weight * node_weight, sparse=True)
        output = tf.concat([mean_out, var_out], axis=1)
        return output

class GGCL_D(Layer):
    """GGCL: the input is distribution"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, bias=False,
                 featureless=False, **kwargs):
        super(GGCL_D, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.dim = int(input_dim / 2)
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_mean'] = glorot([self.dim, output_dim], name='weights_mean')
            self.vars['weights_var'] = glorot([self.dim, output_dim], name='weights_var')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        mean_vector = tf.slice(x, [0, 0], [-1, self.dim])
        var_vector = tf.slice(x, [0, self.dim], [-1, self.dim])
        mean_vector = tf.nn.elu(dot(mean_vector, self.vars['weights_mean']))
        var_vector = tf.nn.relu(dot(var_vector, self.vars['weights_var']))
        node_weight = tf.exp(-var_vector*FLAGS.para_var)
        mean_out = dot(self.support[0], mean_vector * node_weight, sparse=True)
        var_out = dot(self.support[1], var_vector * node_weight * node_weight, sparse=True)
        self.vars['var'] = var_out
        sample_v = tf.random_normal(tf.shape(var_out), 0, 1,
                                    dtype=tf.float32)
        mean_out = mean_out + (tf.math.sqrt(var_out + 1e-8) * sample_v)
        self.vars['mean'] = tf.nn.softmax(mean_out)
        output = mean_out
        return output