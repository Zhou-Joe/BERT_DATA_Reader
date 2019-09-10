
# coding: utf-8

# In[9]:


import cx_Freeze
import sys
import matplotlib
import collections
import pandas
import numpy
import tkinter
import os
import os.path
import scipy
import multiprocessing


PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

base = None
if sys.platform == 'win32':
    base = "WIN32GUI"
executables = [cx_Freeze.Executable("Bert_Data_Reader.py", icon='icon.ico',base=base)]

options = {
    'build_exe': {

        'include_files':[
            os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
            os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),

         ]
    },

}
cx_Freeze.setup(
    name = "Bert Data Reader",
    options = {"build_exe": {"packages":["tkinter","pandas", "scipy","numpy", "os","sys","matplotlib.backends.backend_tkagg","matplotlib.figure","matplotlib.pyplot","scipy.signal","scipy.spatial"]}},
    version = "2.1",
    description = "Bert Data Reader",
    executables = executables
)

