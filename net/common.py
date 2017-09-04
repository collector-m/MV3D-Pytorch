import os
os.environ['HOME'] = '/root'

SEED = 202


TOP_Y_MIN=-20  #40
TOP_Y_MAX=+20
TOP_X_MIN=0
TOP_X_MAX=40  #70.4
TOP_Z_MIN=-2.0    ###<todo> determine the correct values!
TOP_Z_MAX=0.4

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1  #0.1
TOP_Z_DIVISION=0.4

#----------------------------------

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')





#----------------------------------

# std libs
import pickle
from timeit import default_timer as timer
from datetime import datetime
import csv
import pandas as pd
import pickle
import glob

# deep learning libs
import tensorflow as tf
tf.set_random_seed(SEED)

# num libs
import math
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import cv2
import matplotlib.pyplot as plt
#import mayavi.mlab as mlab


# my libs
