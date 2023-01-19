# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:43:55 2021

@author: fhu14
"""

class tst_class:
    def __init__(self):
        self.vars = [1, 2, 3]
    
    def get_vars(self):
        return self.vars
    
    
b = tst_class()
x = b.get_vars()

print(x)
print(b.vars)
print(x is b.vars)

x[1] = 6
print(x)
print(b.vars)
print(x is b.vars)