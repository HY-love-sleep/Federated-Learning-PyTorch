from __future__ import absolute_import
from __future__ import division

import copy
import logging
from collections import Counter
from sys import platform

import copy
import logging
import matplotlib as mpl

if platform == "darwin":  # OS X
    mpl.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.patheffects as PathEffects