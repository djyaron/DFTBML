import numpy as np
try:
    import tensorflow as tf
except:
    pass
import collections
import math
import os
import time
import errno
import logging
from itertools import cycle, islice
import sys

sys.path.insert(0, '../chemtools-webapp/chemtools')

logger = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211383858491185


def format_float(val, prec=3):
    if val == 0.0:
        return '0'
    aval = np.abs(val)
    if aval < 0.01 or aval > 999:
        format_spec = '{0:.' + str(prec) + 'e}'
    else:
        if np.abs(val) < 0.1:
            prec -= int(np.ceil(np.log10(np.abs(val))))
        format_spec = '{0:.' + str(prec) + 'f}'
    return format_spec.format(val)


class UnitManager:
    convs = {'eV': HARTREE_TO_EV, 'kcal/mol': 627.509, 's': 1.0,
             'min': 1.0 / 60.0, ' ': 1.0}

    def __init__(self, units=' '):
        if units not in list(UnitManager.convs.keys()):
            raise ValueError('Unit_manager: unknown unit' + units)
        self._units = units

    def convert(self, val_in_au):
        return val_in_au * UnitManager.convs[self._units]

    def printable(self, val_in_au, prec=3, include_units=True):
        val = self.convert(val_in_au)
        res = format_float(val, prec)
        if include_units:
            res += ' ' + self._units
        return res

    def units(self):
        return self._units


def convert_units(field, val_in_au):
    if 'E' in field:
        units = 'kcal/mol'
    elif field in ['homo', 'lumo']:
        units = 'eV'
    elif field == 'time':
        units = 's'
    else:
        units = ' '
    conv = UnitManager.convs[units]
    return conv * val_in_au, units


def unit_print(field, val_in_au, prec=3, include_units=True):
    if 'E' in field:
        units = 'kcal/mol'
    elif field in ['homo', 'lumo']:
        units = 'eV'
    elif field == 'time':
        units = 's'
    else:
        units = ' '
    conv = UnitManager.convs[units]
    val = conv * val_in_au
    res = format_float(val, prec)
    if include_units:
        res += ' ' + units
    return res


def TriuToSymm(matrix):
    for ind in range(matrix.shape[0]):
        matrix[ind:, ind] = matrix[ind, ind:]
    return matrix


def BMatArr(matrix):
    return np.array(np.bmat(matrix))


def maxabs(mat):
    return np.max(np.abs(mat))


def list_contains(a, b):
    return len(list(set(a).intersection(set(b)))) > 0


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# https://stackoverflow.com/questions/11125212/interleaving-lists-in-python
def roundrobin(iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def rdftb_name(Zs):
    """
    When supplied with a tuple of atomic numbers (one or more), this function
    will return a name

    Arguments:
        Zs (tuple(int)): A tuple containing one or more atomic numbers.

    Returns:
        str: Name of the format a-b, where a and by are the atomic numbers
            involved. Note this will be sorted in order of lowest atomic
            number first. If only a single atomic number is present then
            this will be returned as a string
    """

    if len(Zs) == 2:
        i, j = sorted(Zs)
        return f'rdftb{i}-{j}'
    elif len(Zs) == 1:
        return f'rdftb{Zs[0]}'
    else:
        raise Exception('Warning only 2 or less atomic numbers can be supplied for a spline')


def tfname(prefix, k):
    """
    Generate names for variables, keys, etc. according to its index k.
    """
    res = prefix
    if type(k).__name__ == 'str':
        if len(res) == 0:
            res = k
        else:
            res += '_' + k
    elif type(k).__name__ == 'int':
        res += '_' + str(k)
    elif type(k).__name__ == 'tuple' or type(k).__name__ == 'ndarray':
        if len(res) == 0:
            res = 'size'
        res += '_' + str(k[0])
        for a in k[1:]:
            res += 'x' + str(a)
    return res


def np_segment_sum(data, segment_ids):
    '''
     numpy version of tensorflow's segment_sum
    '''
    max_id = np.max(segment_ids)
    res = np.zeros([max_id + 1], dtype=np.float64)
    for i, val in enumerate(data):
        res[segment_ids[i]] += val
    return res


def map_nested_dicts(ob, func, name=''):
    """
    Convert nested dicts to nested OrderedDicts
    """
    if isinstance(ob, collections.Mapping):
        # return {k: map_nested_dicts(v, func) for k, v in ob.iteritems()}
        res = collections.OrderedDict()
        for k, v in ob.items():
            # print 'mapping for key ', k
            res[k] = map_nested_dicts(v, func, name=tfname(name, k))
        return res
    else:
        if name == '':
            name = 'noname'
        return func(ob, name=name)


def np_to_tf(npdata, tf_category, name='temp'):
    if npdata is None:
        return None
    tf_type = {'float64': tf.float64, 'float32': tf.float64,
               'int64': tf.int32, 'int32': tf.int32}
    if isinstance(npdata, float) or isinstance(npdata, int):
        if isinstance(npdata, float):
            dtype = tf_type['float64']
        else:
            dtype = tf_type['int32']
        if tf_category == 'placeholder':
            res = tf.compat.v1.placeholder(dtype, shape=(), name=name)
        elif tf_category == 'constant':
            res = tf.constant(npdata, dtype=dtype, name=name)
        elif tf_category == 'variable':
            # res = tf.Variable(npdata,dtype = tf_type['float64'], name = name)
            # changing to allow variable scoping to work, see:
            # see https://github.com/tensorflow/tensorflow/issues/1325
            res_initializer = tf.compat.v1.constant_initializer([npdata], dtype=dtype, verify_shape=True)
            # Deprecated variable initialization method in TensorFlow 2
            res = tf.compat.v1.get_variable(name, shape=(), initializer=res_initializer, dtype=dtype)
        else:
            raise ValueError('np_to_tf category ' + tf_category + ' not supported')
        return res

    np_type = npdata.dtype.name
    # print 'np_to_tf called with type ',np_type,' and shape ',npdata.shape
    if tf_category == 'placeholder':
        res = tf.compat.v1.placeholder(tf_type[np_type], shape=npdata.shape, name=name)
    elif tf_category == 'constant':
        res = tf.constant(npdata, dtype=tf_type[np_type], name=name)
    elif tf_category == 'variable':
        # res = tf.Variable(npdata,dtype = tf_type[np_type], name = name)
        # changing to allow variable scoping to work, see:
        # see https://github.com/tensorflow/tensorflow/issues/1325
        res_initializer = tf.compat.v1.constant_initializer(npdata, verify_shape=True)
        res = tf.compat.v1.get_variable(name, shape=npdata.shape, initializer=res_initializer, dtype=tf_type['float64'])
    else:
        raise ValueError('np_to_tf category ' + tf_category + ' not supported')
    return res


def feeddata_recurse(res, tf, np):
    if isinstance(tf, collections.Mapping):
        for k in list(tf.keys()):
            # print 'calling feeddata_recurse with key ', k
            feeddata_recurse(res, tf[k], np[k])
    else:
        # print 'mapping ', tf
        if tf is not None:
            res[tf] = np


def create_feed_dict(fields, tfdata, batch_data):
    res = collections.OrderedDict()
    for field in fields:
        feeddata_recurse(res, tfdata[field], batch_data[field])
    return res


def get_values_from_nested_dicts(data, key, result):
    ''' 
    Recurse through nested dicts, and pull out value for key. If key is
    found, the dict is considered to be the final leaf
    '''
    if isinstance(data, list):
        for x in data:
            get_values_from_nested_dicts(x, key, result)
    elif isinstance(data, dict):
        if key in list(data.keys()):
            result.append(data[key])
        else:
            for v in list(data.values()):
                get_values_from_nested_dicts(v, key, result)


def layer_type(val):
    '''
      if val is int:   return ('r', val) indicating relu layer
      if val is  str of form 'str-#' returns (str, int)
    '''
    if type(val) is int:
        return ('r', val)
    elif type(val) is str:
        p1 = val.partition('-')
        if len(p1) != 3 or not p1[2].isdigit():
            raise ValueError('util.py: layer_type not recognized: ' + str(val))
        layer_type = p1[0]
        layer_size = int(p1[2])
        return (layer_type, layer_size)
    else:
        raise ValueError('util.py: layer_type not recognized: ' + str(val))


def feedforward_NN(name_scope, features, hidden_layers):
    """ Based on https://github.com/tensorflow/tensorflow/blob/r0.11/
          tensorflow/examples/tutorials/mnist/mnist.py
    Args:
     features: features placeholder (nsamples, nfeatures)
     hidden_units: List with size of the hidden layers 
    Returns:
     elements: tensor of shape (nsamples) holding element predictions
    """
    layer_types = [layer_type(x) for x in hidden_layers]
    hidden_types = [x[0] for x in layer_types]
    hidden_units = [x[1] for x in layer_types]
    nfeatures = features.get_shape()[1].value
    # holds sizes needed to build the layers
    size_vector = [nfeatures] + hidden_units
    current_input = features
    for ihidden in range(len(hidden_units)):
        with tf.variable_scope(name_scope + 'hidden' + str(ihidden)):
            # weights = tf.Variable(
            # tf.truncated_normal(size_vector[ihidden:(ihidden+2)],
            #        stddev=1.0 / math.sqrt(float(size_vector[ihidden])),
            #                    dtype=tf.float64, name='weights'))
            weights = tf.get_variable('W',
                                      shape=size_vector[ihidden:(ihidden + 2)],
                                      dtype=tf.float64,
                                      initializer=tf.contrib.layers.xavier_initializer(
                                          uniform=True))
            # biases = tf.Variable(tf.zeros([size_vector[ihidden+1]],dtype=tf.float64),
            #             name='B', dtype = tf.float64)
            biases = tf.get_variable('B', shape=[size_vector[ihidden + 1]],
                                     dtype=tf.float64,
                                     initializer=tf.zeros_initializer(dtype=tf.float64))
            outs = tf.matmul(current_input, weights) + biases
            ltype = hidden_types[ihidden]
            if ltype == 'r':
                hidden = tf.nn.relu(outs, name='Vals')
            elif ltype == 'r6':
                hidden = tf.nn.relu6(outs, name='Vals')
            elif ltype == 's':
                hidden = tf.nn.sigmoid(outs, name='Vals')
            elif ltype == 't':
                hidden = tf.nn.tanh(outs, name='Vals')
            elif ltype == 'e':
                hidden = tf.nn.elu(outs, name='Vals')
            else:
                raise ValueError('util.py ffnet hidden_type not recognized: '
                                 + str(ltype))

        current_input = hidden
    return hidden


def final_layer(name_scope, features):
    # Final layer is Linear
    # nsamples = features.get_shape()[0].value
    nfeatures = features.get_shape()[1].value
    with tf.variable_scope(name_scope + 'final'):
        init1 = tf.truncated_normal_initializer(
            stddev=1.0 / math.sqrt(float(nfeatures)), dtype=tf.float64)
        weights = tf.get_variable('weights', shape=[nfeatures, 1],
                                  dtype=tf.float64, initializer=init1)
        biases = tf.get_variable('biases', shape=[1],
                                 dtype=tf.float64,
                                 initializer=tf.zeros_initializer(dtype=tf.float64))
        elements = tf.reshape(tf.matmul(features, weights) + biases, [-1],
                              name='Vals')
    return elements


# from: http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    # Could perhaps use "os.path.exists" instead?
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def print_time(tcurr, msg):
    tnew = time.time()
    logger.info(msg + " : " + str(tnew - tcurr))
    return tnew


class UpdateManager:
    def __init__(self, current_step=0, dx=None, min_steps=None,
                 min_time=None, logic='and', dx_compare='gt'):
        self.dx = dx
        self.dx_compare = dx_compare
        self.min_steps = min_steps
        self.last_steps = current_step
        self.min_time = min_time
        self.last_time = time.time()
        self.logic = logic

    def should_update(self, istep, dx=None):
        tests = []
        if self.dx is not None and dx is not None:
            if self.dx_compare == 'gt':
                tests.append(dx >= self.dx)
            else:
                tests.append(dx <= self.dx)
        if self.min_steps is not None:
            tests.append((istep - self.last_steps) >= self.min_steps)
        if self.min_time is not None:
            tests.append((time.time() - self.last_time) >= self.min_time)
        if self.logic == 'and':
            res = all(tests)
        else:
            res = any(tests)
        if res:
            self.last_steps = istep
            if self.min_time is not None:
                self.last_time = time.time()
        return res

    def reset(self, current_step=0):
        self.last_steps = current_step
        if self.min_time is not None:
            self.last_time = time.time()

    def __str__(self):
        res = ''
        if self.dx is not None:
            res += ' dx = ' + str(self.dx)
        if self.min_steps is not None:
            res += ' min steps = ' + str(self.min_steps)
        if self.min_time is not None:
            res += ' min time = ' + str(self.min_time)
        return res

    def prompt_user(self):
        res = ''
        if self.dx is not None:
            x = eval(input('change dx from ' + str(self.dx) + ' to ? '))
            self.dx = x
            res += ' dx changed to ' + str(self.dx)
        if self.min_steps is not None:
            x = eval(input('change min steps from ' + str(self.min_steps) + ' to ? '))
            self.min_steps = x
            res += ' min steps changed to ' + str(self.min_steps)
        if self.min_time is not None:
            print('current time is ', end=' ')
            x = eval(input('change min time from ' + str(self.min_time) + ' to ? '))
            self.min_time = x
            res += ' min time changed to ' + str(self.min_time)
            self.last_time = time.time()
        return res


def format_multiline(strings, linewidth=80):
    full_msg = ''
    line = ''
    for x in strings:
        if len(line) + len(x) < linewidth:
            line += ' ' + x
        else:
            full_msg += 'n' + line
            line = x
    if len(line) > 0:
        full_msg += '\n' + line
    return full_msg
