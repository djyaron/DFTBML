# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:38:07 2021

@author: fhu14
"""
#%% Imports, definitions
import numpy as np
import torch
Tensor = torch.Tensor
from typing import List, Union, Dict
import collections
import random
import os

#%% Code behind

def np_segment_sum(data, segment_ids):
    '''
     numpy version of tensorflow's segment_sum
    '''
    max_id = np.max(segment_ids)
    res = np.zeros([max_id + 1], dtype=np.float64)
    for i, val in enumerate(data):
        res[segment_ids[i]] += val
    return res

def torch_segment_sum(data: Tensor, segment_ids: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor: 
    r"""Function for summing elements together based on index
    
    Arguments:
        data (Tensor): The data to sum together
        segment_ids (Tensor): The indices used to sum together corresponding elements
        device (torch.device): The device to execute the operations on (CPU vs GPU)
        dtype (torch.dtype): The datatype for the result
    
    Returns:
        res (Tensor): The resulting tensor from executing the segment sum
    
    Notes: This is similar to scatter_add for PyTorch, but this is easier to deal with.
        The segment_ids, since they are being treated as indices, must be a tensor
        of integers
    """
    max_id = torch.max(segment_ids)
    res = torch.zeros([max_id + 1], device = device, dtype = dtype)
    res = res.scatter_add(0, segment_ids.long(), data)
    return res

def maxabs(mat):
    return np.max(np.abs(mat))

def list_contains(a, b):
    return len(list(set(a).intersection(set(b)))) > 0

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

def recursive_type_conversion(data: Dict, ignore_keys: List[str], device: torch.device = None, 
                              dtype: torch.dtype = torch.double, grad_requires: bool = False) -> None:
    r"""Performs destructive conversion of elements in data from np arrays to Tensors
    
    Arguments:
        data (Dict): The dictionary to perform the recursive type conversion on
        ignore_keys (List[str]): The list of keys to ignore when doing
            the recursive type conversion
        device (torch.device): Which device to put the tensors on (CPU vs GPU).
            Defaults to None.
        dtype (torch.dtype): The datatype for all created tensors. Defaults to torch.double
        grad_requires (bool): Whether or not created tensors should have their 
            gradients enabled. Defaults to False
    
    Returns:
        None
    
    Notes: None
    """
    for key in data:
        if key not in ignore_keys:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.tensor(data[key], dtype = dtype, device = device)            
            elif isinstance(data[key], collections.OrderedDict) or isinstance(data[key], dict):
                recursive_type_conversion(data[key], ignore_keys, device = device, dtype = dtype)
                
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

def create_split_mapping(s) -> Dict:
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