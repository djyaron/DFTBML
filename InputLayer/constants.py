# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:55:06 2021

@author: fhu14
"""

from collections import namedtuple
Model = namedtuple('Model',['oper', 'Zs', 'orb'])
RawData = namedtuple('RawData',['index','glabel','Zs','atoms','oper','orb','dftb','rdist'])