# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:26:54 2021

@author: fhu14
"""

from collections import namedtuple
Model = namedtuple('Model',['oper', 'Zs', 'orb'])
RawData = namedtuple('RawData',['index','glabel','Zs','atoms','oper','orb','dftb','rdist'])