"""
Converts dictionaries from h5 files and back.
"""

import numpy as np
import h5py
import os

class h5_handler(object):

    @classmethod
    def save_dict_new_h5(cls, dic: dict, dest: str):
        r"""Entry point to turning a dictionary to an h5 file
        
        Arguments:
        dic: The dictionary to be saved to an h5 file.
        dest: The name and path for the new h5 file to be saved to.
        """
        if os.path.exists(dest):
            print("This h5 file already exists.")
        else:
            with h5py.File(dest, 'w') as file:
                cls.recursive_h5_dict_save(file, '/', dic)


    @classmethod
    def recursive_h5_dict_save(cls, h5file, path: str, dic: dict):
        r"""Recursively goes through a given dictionary and translates it into an h5 file.
        
        Arguments:
        h5file: The file object for the h5 file being created.
        path: A path within the h5file to continue saving.
        dic: The dictionary to be saved to an h5 file.
        """
        for keypath, item in dic.items():
            if isinstance(item, (np.int64, np.float64, str)):
                h5file[path + keypath] = item
            elif isinstance(item, (np.ndarray, list)):
                h5file[path + keypath] = item
            elif isinstance(item, dict):
                cls.recursive_h5_dict_save(h5file, path+keypath+'/', item)
            else:
                raise ValueError(f"Can't save type: {type(item)}")
    
    @classmethod
    def h5_to_dict(cls, file: str, path = '/'):
        r"""Entry point to turning an h5 file to a dictionary.
        
        Arguments:
        file: The path to the h5 file to load.
        path: The path within the h5file to load from.

        Returns:
        The created dictionary
        """
        with h5py.File(file, 'r') as h5file:
            return cls.recursive_h5_to_dict(h5file, path)

    @classmethod
    def recursive_h5_to_dict(cls, h5file, path):
        r"""Recursively goes through the dict to convert it to h5.
        
        Arguments:
        h5file: The file object of the h5 file to load.
        path: The path within the h5file to continue loading from.

        Returns:
        return_dict: A dictionary copy of the h5 file contents 
        """    
        return_dict = {}
        if path == '/':
            h5filestart = h5file
        else:
            h5filestart = h5file[path]
        for keypath, item in h5filestart.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                if isinstance(type(item[()]), bytes):
                    return_dict[keypath] = item[()].decode('UTF-8')
                else:
                    return_dict[keypath] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                return_dict[keypath] = cls.recursive_h5_to_dict(h5file, path+keypath+'/')
        return return_dict
