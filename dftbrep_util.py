import numpy as np
import numpy.linalg as la
import os
import pickle as pkl
from _pickle import UnpicklingError
from h5py import File
from hashlib import sha256
from time import time


class Timer:
    """
    To time a code block, use
    with Timer(prompt):
        CODE_BLOCK
    """

    def __init__(self, prompt=None):
        self.prompt = prompt

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        self.seconds = self.end - self.start
        if self.prompt:
            print(f"Wall time of {self.prompt}: {self.seconds:.1f} seconds")
        else:
            print(f"Wall time: {self.seconds:.1f} seconds")


def get_sha256(file_path):
    BLOCK_SIZE = 65536  # The size of each read from the file

    file_hash = sha256()  # Create the hash object, can use something other than `.sha256()` if you wish
    with open(file_path, 'rb') as f:  # Open the file to read it's bytes
        fb = f.read(BLOCK_SIZE)  # Read from the file. Take in the amount declared above
        while len(fb) > 0:  # While there is still data being read from the file
            file_hash.update(fb)  # Update the hash
            fb = f.read(BLOCK_SIZE)  # Read the next block from the file

    return file_hash.hexdigest()  # Return the hexadecimal digest of the hash


def path_check(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_dataset_type(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not exists")

    ext = os.path.splitext(dataset_path)[-1]
    if ext == '.h5':
        try:
            with File(dataset_path, 'r') as dataset:
                dataset_type = "h5"
        except OSError:
            raise TypeError(f"{dataset_path} is not a valid HDF5 dataset")
    elif ext in ('.pkl', '.pickle', '.p'):
        try:
            with open(dataset_path, 'rb') as f:
                dataset = pkl.load(f)
                dataset_type = "pkl"
                del dataset
        except UnpicklingError:
            raise TypeError(f"{dataset_path} is not a valid pickle dataset")
    else:
        raise NotImplementedError(f"{dataset_path} is not a supported dataset")

    return dataset_type

