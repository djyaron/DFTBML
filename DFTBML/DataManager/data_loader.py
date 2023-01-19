# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:23:31 2021

@author: fhu14

This class is only used during the precompute process
"""
#%% Imports, definitions
from typing import List, Dict
import random

#%% Code behind

class data_loader:

    def __init__(self, dataset: List[Dict], batch_size: int, shuffle: bool = True, 
                 shuffle_scheme: str = 'random') -> None:
        r"""Initializes the data_loader object for batching data.
        
        Arguments:
            dataset (List[Dict]): A list of dictionaries containing information for
                all the molecules, with each molecule represented as a dictionary.
            batch_size (int): The number of molecules to have per batch
            shuffle (bool): Whether or not to shuffle batches. Defaults to True
            shuffle_scheme (str): The scheme for shuffling. Defaults to "random"
        
        Returns:
            None
        
        Notes: This is a very simple data_loader implementation that uses sequential
            batching on a list of data. Once all the data has been iterated over, the
            loader can be re-iterated over and it will shuffle the batches between iterations.
            
            This loader can be used for any list-type dataset, but for this use case we 
            specify List[Dict] since that is the data format being used here.
        """
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_method = shuffle_scheme
        self.batch_creation()
        self.batch_index = 0
    
    def create_batches(self, data: List[Dict]) -> List[List[Dict]]:
        r"""Generates the batches from the list of data.
        
        Arguments:
            data (List[Dict]): List of molecule dictionaries from which to generate
                the batches
        
        Returns:
            batches (List[List[Dict]]): A list of lists of molecule dictionaries, where
                each inner list represents one batch
                
        Notes: If the number of elements in the data is not a multiple of the batch size,
            the last batch will have less elements than the batch size.
        """
        batches = list()
        for i in range(0, len(data), self.batch_size):
            current_batch = data[i : i + self.batch_size]
            batches.append(current_batch)
        return batches
    
    def batch_creation(self) -> None:
        r"""Wrapper method for creating the batches
        
        Arguments:
            None
        
        Returns: 
            None
        
        Notes: Initializes the batches by calling self.create_batches, and 
            also initializes the batch index used for iterating over the 
            data_loader
        """
        self.batches = self.create_batches(self.data)
        self.batch_index = 0
        
    def shuffle_batches(self) -> None:
        r"""Shuffles batches and resets the batch_index
        
        Arguments: 
            None
            
        Returns:
            None
        
        Notes: Called at end of iteration over data_loader
        """
        random.shuffle(self.batches)
        self.batch_index = 0
        
    def shuffle_total_data(self) -> None:
        r"""Reshuffles the original data from which the batches are generated
        
        Arguments:
            None
        
        Returns:
            None
        
        Notes: None
        """
        random.shuffle(self.data)
        self.batch_creation()
    
    def __iter__(self):
        r"""
        Method for treating the data_loader as the iterator.
        """
        return self
    
    def __next__(self) -> List[Dict]:
        r"""Method for retrieving next element in iterator
        
        Arguments:
            None
        
        Returns:
            return_batch (List[Dict]): The current batch to be used
        
        Notes: None
        
        Raises:
            StopIteration: If iteration over all batches in data_loader is complete
        """
        if self.batch_index < len(self.batches):
            return_batch = self.batches[self.batch_index]
            self.batch_index += 1
            return return_batch
        else:
            # Automatically shuffle the batches after a full iteration
            self.shuffle_batches()
            raise StopIteration