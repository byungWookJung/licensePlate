#-*- coding: utf-8 -*-
# hadle_plate.py

import cv2
import numpy as np
import os

from license_plate_lib import detect_chars
from license_plate_lib import detect_plates
from license_plate_lib import possible_plate
import imutils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

# showSteps = True
showSteps = False

###################################################################################################
'''
    make the image histogram equalized
'''
def makeHistogramEqualization(imgOriginalScene):

    # histogram 洹쒖씪�솕 (histogram equalization)
    hist, bins = np.histogram(imgOriginalScene.ravel(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    imgOriginalScene = cdf[imgOriginalScene]

    return imgOriginalScene
# end function




###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
# TODO : 삭제
def drawRedRectangleAroundPlate2(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_YELLOW, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_YELLOW, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_YELLOW, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_YELLOW, 2)
# end function
# end of TODO : 삭제

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    # 한글 폰트 설정
    font_location = "C:\Windows\Fonts\H2HDRM.TTF"
    font_name = fm.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)

    plt.title("자동차 번호 : " + licPlate.strChars)
    plt.imshow(imgOriginalScene)
    plt.show()

    # ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    # ptCenterOfTextAreaY = 0
    #
    # ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    # ptLowerLeftTextOriginY = 0
    #
    # sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    # plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape
    #
    # intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    # fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    # intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale
    #
    # textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize
    #
    #         # unpack roatated rect into center point, width and height, and angle
    # ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene
    #
    # intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    # intPlateCenterY = int(intPlateCenterY)
    #
    # ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate
    #
    # if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
    #     ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    # else:                                                                                       # else if the license plate is in the lower 1/4 of the image
    #     ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # # end if
    #
    # textSizeWidth, textSizeHeight = textSize                # unpack text size width and height
    #
    # ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    # ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height
    #
    # # write the text on the image
    # cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)

# end function


















