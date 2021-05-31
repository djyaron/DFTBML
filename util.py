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

from typing import Optional
from hashlib import sha256
import matplotlib as mpl
from _pickle import UnpicklingError
from h5py import File
import pickle as pkl
from collections import Counter

from typing import Union, List, Dict
import torch
import random

import matplotlib.pyplot as plt
from functools import reduce
import numpy as np

#sys.path.insert(0, '../chemtools-webapp/chemtools')

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

#%% Functions for repulsive utility

class Timer:

    def __init__(self, prompt: Optional[str] = None) -> None:
        r"""Print the wall time of the execution of a code block

        Args:
            prompt (str): Specify the prompt in the output. Optional.

        Returns:
            None

        Examples:
            >>> from util import Timer
            >>> with Timer("TEST"):
            ...     # CODE_BLOCK
            Wall time of TEST: 0.0 seconds
            >>> from util import Timer
            >>> with Timer():
            ...     # CODE_BLOCK
            Wall time: 0.0 seconds
        """
        self.prompt = prompt

    def __enter__(self):
        self.start = time()  # <-- Record the time when Timer is called
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end = time()  # <-- Record the time when Timer exits
        self.seconds = self.end - self.start  # <-- Compute the time interval
        if self.prompt:
            print(f"Wall time of {self.prompt}: {self.seconds:.1f} seconds")
        else:
            print(f"Wall time: {self.seconds:.1f} seconds")


def count_n_heavy_atoms(atomic_numbers):
    counts = sum([c for a, c in dict(Counter(atomic_numbers)).items() if a > 1])
    return counts


def path_check(file_path: str) -> None:
    r"""Check if the specified path exists, if not then create the parent directories recursively

    Args:
        file_path: Path to be checked

    Returns:
        None

    Examples:
        >>> from util import path_check
        >>> file_path_1 = "EXISTING_DIR/NOT_EXISTING_FILE"
        >>> path_check(file_path_1)
        # No output
        >>> file_path_2 = "NOT_EXISTING_DIR/NOT_EXISTING_FILE"
        >>> path_check(file_path_2)
        # "/NOT_EXISTING_DIR" is created
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  #: Recursively create the parent directories of the file


def get_dataset_type(dataset_path: str) -> str:
    r"""Check the dataset type (HDF5 or pickle) and the validity of the dataset

    Args:
        dataset_path: Path to the dataset

    Returns:
        dataset_type: "h5" or "pkl"

    Examples:
        >>> from util import get_dataset_type
        >>> dataset_path_1 = "DIR/valid_dataset.h5"
        >>> get_dataset_type(dataset_path_1)
        'h5'
        >>> dataset_path_2 = "DIR/corrupted_dataset.pkl"
        >>> get_dataset_type(dataset_path_2)
        ValueError: DIR/corrupted_dataset.pkl is not a valid pickle dataset

    Raises:
        FileNotFoundError: if the dataset does not exist
        ValueError: if dataset cannot be opened
        NotImplementedError: if the extension of the dataset is not one of the following
                             '.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5',
                             '.pkl', '.pickle', '.p'
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not exists")

    ext = os.path.splitext(dataset_path)[-1].lower()
    if ext in ('.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5'):
        try:
            with File(dataset_path, 'r'):
                dataset_type = 'h5'
        except OSError:
            raise ValueError(f"{dataset_path} is not a valid HDF5 dataset")
    elif ext in ('.pkl', '.pickle', '.p'):
        try:
            with open(dataset_path, 'rb') as f:
                pkl.load(f)
                dataset_type = 'pkl'
        except UnpicklingError:
            raise ValueError(f"{dataset_path} is not a valid pickle dataset")
    else:
        raise NotImplementedError(f"{dataset_path} is not a supported dataset")

    return dataset_type


def mpl_default_setting():
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['figure.subplot.hspace'] = 0.3
    mpl.rcParams['figure.subplot.wspace'] = 0.3
    mpl.rcParams['figure.titlesize'] = 'large'

    mpl.rcParams['font.family'] = ['Arial']
    mpl.rcParams['font.size'] = 16

    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.loc'] = 'upper right'

    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 'large'
    mpl.rcParams['xtick.major.size'] = 0
    mpl.rcParams['xtick.minor.size'] = 0
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['ytick.labelsize'] = 'large'
    mpl.rcParams['ytick.major.size'] = 0
    mpl.rcParams['ytick.minor.size'] = 0

    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'png'
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.bbox'] = 'tight'

#%% Other utilities

class Settings:
    def __init__(self, settings_dict: Dict) -> None:
        r"""Generates a Settings object from the given dictionary
        
        Arguments:
            settings_dict (Dict): Dictionary containing key value pairs for the
                current hyperparmeter settings
        
        Returns:
            None
        
        Notes: Using an object rather than a dictionary is easier since you can
            just do settings.ZZZ rather than doing the bracket notation and the quotes.
        """
        for key in settings_dict:
            setattr(self, key, settings_dict[key])

def apx_equal(x: Union[float, int], y: Union[float, int], tol: float = 1e-12) -> bool:
    r"""Compares two floating point numbers for equality with a given threshold
    
    Arguments:
        x (float): The first number to be compared
        y (float): The second number to be compared
        
    Returns:
        equality (bool): Whether the two given numbers x and y are equal
            within the specified threshold by comparing the absolute value
            of their difference.
            
    Notes: This method works with both integers and floats, which are the two 
        numeric types. Chosen workaround for determining float equality

    """
    return abs(x - y) < tol

def update_pytorch_arguments(settings: Settings) -> None:
    r"""Updates the arguments in the settings object to the corresponding 
        PyTorch types
        
    Arguments:
        settings (Settings): The settings object representing the current set of 
            hyperparameters
    
    Returns:
        None
        
    Notes: First checks if a CUDA-capable GPU is available. If not, it will
        default to using CPU only.
    """
    if settings.tensor_dtype == 'single':
        print("Tensor datatype set as single precision (float 32)")
        settings.tensor_dtype = torch.float
    elif settings.tensor_dtype == 'double':
        print("Tensor datatype set as double precision (float 64)")
        settings.tensor_dtype = torch.double
    else:
        raise ValueError("Unrecognized tensor datatype")
        
    num_gpus = torch.cuda.device_count()
    if settings.tensor_device == 'cpu':
        print("Tensor device set as cpu")
        settings.tensor_device = 'cpu'
    elif num_gpus == 0 or (not (torch.cuda.is_available())):
        print("Tensor device set as cpu because no GPUs are available")
        settings.tensor_device = 'cpu'
    else:
        gpu_index = int(settings.device_index)
        if gpu_index >= num_gpus:
            print("Invalid GPU index, defaulting to CPU")
            settings.tensor_device = 'cpu'
        else:
            print("Valid GPU index, setting tensor device to GPU")
            #I think the generic way is to do "cuda:{device index}", but not sure about this
            settings.tensor_device = f"cuda:{gpu_index}"
            print(f"Used GPU name: {torch.cuda.get_device_name(settings.tensor_device)}")

def construct_final_settings_dict(settings_dict: Dict, default_dict: Dict) -> Dict:
    r"""Generates the final settings dictionary based on the input settings file and the
        defaults file
    
    Arguments:
        settings_dict (Dict): Dictionary of the user-defined hyperparameter settings. 
            Read from the settings json file
        default_dict (Dict): Dictionary of default hyperparameter settings. 
            Read from the defaults json file
    
    Returns:
        final_settings_dict (Dict): The final dictionary containing the settings
            to be used for the given run over all hyperparameters
    
    Notes: If something is not specified in the settings file, then the
        default value is pulled from the default_dict to fill in. For this reason,
        the default_dict contains the important keys and the settings file
        can contain a subset of these important keys. The settings file will include
        some information that is not found in the default dictionary, such as the 
        name given to the current run and the directory for saving the skf files at the end
        
        The settings_dict must contain the run_name key
    """
    final_dict = dict()
    for key in default_dict:
        if key not in settings_dict:
            final_dict[key] = default_dict[key]
        else:
            final_dict[key] = settings_dict[key]
    try:
        final_dict['run_id'] = settings_dict['run_id']
    except:
        raise KeyError("Settings file must include the 'run_id' field!")
    
    return final_dict

def dictionary_tuple_correction(input_dict: Dict) -> Dict:
    r"""Performs a correction on the input_dict to convert from string to tuple
    
    Arguments:
        input_dict (Dict): The dictionary that needs correction
    
    Returns:
        new_dict (Dict): Dictionary with the necessary corrections
            applied
    
    Notes: The two dictionaries that need correction are the cutoff dictionary and the
        range correction dictionary for model_range_dict. For the dictionary used to 
        correct model ranges, the keys are of the form "elem1,elem2" where elem1 and elem2 
        are the atomic numbers of the first and second element, respectively. These
        need to be converted to a tuple of the form (elem1, elem2).
        
        For the dictionary used to specify cutoffs (if one is provided), the format 
        of the keys is "oper,elem1,elem2" where oper is the operator of interest and
        elem1 and elem2 are again the atomic numbers of the elements of interest. This
        will be converted to a tuple of the form (oper, (elem1, elem2)). A check is 
        performed between these cases depending on the number of commas.
        
        The reason for this workaround is because JSON does not support tuples. 
        An alternative would have been to use a string representation of the tuple
        with the eval() method. 
    """
    num_commas = list(input_dict.keys())[0].count(",")
    if num_commas == 0:
        print("Dictionary does not need correction")
        print(input_dict)
        return input_dict #No correction needed
    new_dict = dict()
    #Assert key consistency in the dictionary
    for key in input_dict:
        assert(key.count(",") == num_commas)
        key_splt = key.split(",")
        if len(key_splt) == 2:
            elem1, elem2 = int(key_splt[0]), int(key_splt[1])
            new_dict[(elem1, elem2)] = input_dict[key]
        elif len(key_splt) == 3:
            oper, elem1, elem2 = key_splt[0], int(key_splt[1]), int(key_splt[2])
            new_dict[(oper, (elem1, elem2))] = input_dict[key]
        else:
            raise ValueError("Given dictionary does not need tuple correction!")
    return new_dict

def paired_shuffle(lst_1: List, lst_2: List) -> (list, list):
    r"""Shuffles two lists while maintaining element-wise corresponding ordering
    
    Arguments:
        lst_1 (List): The first list to shuffle
        lst_2 (List): The second list to shuffle
    
    Returns:
        lst_1 (List): THe first list shuffled
        lst_2 (List): The second list shuffled
    """
    temp = list(zip(lst_1, lst_2))
    random.shuffle(temp)
    lst_1, lst_2 = zip(*temp)
    lst_1, lst_2 = list(lst_1), list(lst_2)
    return lst_1, lst_2

def convert_key_to_num(elem: Dict) -> Dict:
    return {int(k) : v for (k, v) in elem.items()}

def create_split_mapping(s: Settings) -> Dict:
    r"""Creates a fold mapping for the case where individual folds are
        combined to create total training/validation data
    
    Arguments:
        s (Settings): The Settings object with hyperparameter values
    
    Returns:
        mapping (Dict): The dictionary mapping current fold number to the 
            numbers of individual folds for train and validate. This only applies
            when you are combining individual folds. Each entry in the dictionary
            contains a list of two lists, the first inner list is the fold numbers 
            for training and the second inner list is the fold numbers for validation.
    
    Notes: Suppose we are training on five different folds / blocks of data numbered
        1 -> N. In the first training iteration in a CV driver mode, if the cv_mode is 
        'normal', we will train on the combined data of N - 1 folds together and test 
        on the remaining Nth fold. If the cv_mode is 'reverse', we will validate 
        on N - 1 folds while training on the remaining Nth fold. In previous iterations,
        each fold really contained a training set of validation set of feed dictionaries;
        now each fold means just one set of feed dictionaries, and we have to use all folds
        for every iteration of CV. 
    """
    num_directories = len(list(filter(lambda x : '.' not in x, os.listdir(s.top_level_fold_path))))
    num_folds = s.num_folds
    #num_folds should equal num_directories
    assert(num_folds == num_directories)
    cv_mode = s.cv_mode
    full_fold_nums = [i for i in range(num_folds)]
    mapping = dict()
    for i in range(num_folds):
        mapping[i] = [[],[]]
        if cv_mode == 'normal':
            mapping[i][1].append(i)
            mapping[i][0] = full_fold_nums[0 : i] + full_fold_nums[i + 1:]
        elif cv_mode == 'reverse':
            mapping[i][0].append(i)
            mapping[i][1] = full_fold_nums[0 : i] + full_fold_nums[i + 1:]
    return mapping

def visualize_loss_tracker(lt: Dict, train_color: str = 'b', valid_color: str = 'g') -> None:
    r"""Takes in a loss_tracker from a training run and visualizes the
        training and validation losses on the same graph
    
    Arguments:
        lt (Dict): The loss tracker dictionary containing the train and validation
            losses for all targets of interest
        train_color (str): String representing the color for plotted training
            data. Defaults to blue, 'b'
        valid_color (str): String representing the color for plotted validation 
            data. Defaults to green, 'g'
    
    Returns:
        None
    """
    for target in lt:
        valid_vals, train_vals = lt[target][0], lt[target][1]
        assert(len(valid_vals) == len(train_vals))
        epochs = [i for i in range(len(valid_vals))]
        fig, axs = plt.subplots()
        axs.plot(epochs, valid_vals, label = 'validation', color = valid_color)
        axs.plot(epochs, train_vals, label = 'training', color = train_color)
        axs.set_ylabel('Average epoch loss')
        axs.set_xlabel('Epoch')
        axs.set_title(f"{target} train and validation loss")
        axs.legend()
        plt.show()
        
def find_min_max_heavy(feeds: List[Dict]) -> (int, int):
    r"""Finds the minimum and maximum numbers of heavy atoms in molecules
        in a feed.
    
    Arguments:
        feeds (list[Dict]): The list of feeds to check
    
    Returns:
        (int, int): The minimum and maximum number of heavy atoms
    
    Notes: The feeds must all have the 'nheavy' key contained within.
    """
    heavies = [list(feed['nheavy'].values()) for feed in feeds]
    heavies = heavies = list(reduce(lambda x, y : x + y, heavies))
    heavies = np.concatenate(heavies)
    return min(heavies), max(heavies)
    
        
