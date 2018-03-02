#-*- coding: utf-8 -*-
# PossiblePlate.py

import cv2
import numpy as np

###################################################################################################
class PossiblePlate:

    # constructor #################################################################################
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""
        self.numDigits = 0
    # end constructor

# end class




