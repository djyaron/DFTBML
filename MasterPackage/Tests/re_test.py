# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 21:45:58 2021

@author: fhu14
"""
"""
A test for regular expression parsing of charges and dipoles
"""

#%% Imports, definitions
import re
from MasterConstants import gross_charge, cm5_charge, dipole_line

#%% Code behind
def test_charge_parsing() -> None:
    r"""Tests the accuracy of the patterns derived for regular expressions
    """
    
    gross_matcher = re.compile(gross_charge)
    cm5_matcher = re.compile(cm5_charge)
    
    tst_file_name = "test_files/re_detailed_tst.out"
    content = open(tst_file_name, 'r').read()
    
    gross_correct = ''' Atomic gross charges (e)
 Atom           Charge
    1       0.53305546
    2       0.27503914
    3       0.27401775
    4       0.30980044
    5      -0.47512952
    6      -0.05691436
    7      -0.04346443
    8      -0.07222214
    9      -0.31664229
   10      -0.42754006'''
   
    cm5_correct = ''' CM5 corrected atomic gross charges (e)
 Atom           Charge
    1       0.71072551
    2       0.52062343
    3       0.47392195
    4       0.53191256
    5      -0.88705963
    6      -0.16635063
    7      -0.24647650
    8      -0.09669764
    9      -0.38347283
   10      -0.45712621'''
    
    gross_result = gross_matcher.search(content)
    cm5_result = cm5_matcher.search(content)
    
    gross_start, gross_end = gross_result.span()
    cm5_start, cm5_end = cm5_result.span()
    
    gross_str_exp = gross_result.string[gross_start : gross_end]
    cm5_str_exp = cm5_result.string[cm5_start : cm5_end]
    
    gross_str_exp = gross_str_exp.replace("\n", "").replace(" ", "")
    cm5_str_exp = cm5_str_exp.replace("\n", "").replace(" ", "")
    
    gross_str_true = gross_correct.replace("\n", "").replace(" ", "")
    cm5_str_true = cm5_correct.replace("\n", "").replace(" ", "")
    
    assert(gross_str_exp == gross_str_true)
    assert(cm5_str_exp == cm5_str_true)
    
    print("Regex parsing for charges (gross and cm5) passed")

def test_dipole_parsing() -> None:
    r"""Tests the dipole parsing capability of the regex expression developed
    """
    au_dipole_true = "Dipole moment:   -0.93224624   -2.01538010    0.05349291 au\n"
    debye_dipole_true = "Dipole moment:   -2.36953379   -5.12258567    0.13596542 Debye\n"
    
    dipole_matcher = re.compile(dipole_line)
    tst_file_name = "test_files/re_detailed_tst.out"
    content = open(tst_file_name, 'r').read()
    dipoles = dipole_matcher.findall(content)
    assert(len(dipoles) == 2)
    assert(dipoles[0] == au_dipole_true) #au dipole comes in first
    assert(dipoles[1] == debye_dipole_true)
    
    print("Dipole parsing test passed")
    

def run_re_tests():
    test_charge_parsing()
    test_dipole_parsing()
    
#%% Main block

if __name__ == "__main__":
    run_re_tests()
   


