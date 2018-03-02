#-*- coding: utf-8 -*-
# DetectPlates.py

import cv2
import numpy as np
import math
import random

from license_plate_lib import handle_plate
from license_plate_lib import preprocess
from license_plate_lib import detect_chars
from license_plate_lib import possible_plate
from license_plate_lib import possible_char
import imutils
from license_plate_lib.license_plate import LicensePlateDetector
from license_plate_lib.descriptors import BlockBinaryPixelSum

# module level variables ##########################################################################
#PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_WIDTH_PADDING_FACTOR = 1.4
PLATE_HEIGHT_PADDING_FACTOR = 1.5

##########################################################################################
# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

###################################################################################################
'''
    make the list of possible license plate candidates
'''
def detectPlatesInScene(digitetcModel, imgOriginalScene):
    listOfPossiblePlates = []                   # this will be the return value

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if handle_plate.showSteps == True: # show steps #######################################################
        cv2.imshow("0", imgOriginalScene)
    # end if # show steps #########################################################################

    # preprocess to get grayscale and threshold images
    imgGrayscaleScene, imgThreshScene = preprocess.preprocess(imgOriginalScene)

    if handle_plate.showSteps == True: # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if # show steps #########################################################################

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    # 1차 문자후모 목록 구성
    listOfPossibleCharsInScene = findPossibleCharsInScene(digitetcModel, imgThreshScene)

    if handle_plate.showSteps == True: # show steps #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))         # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, handle_plate.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if # show steps #########################################################################

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    # 2차 문자후모 목록 구성
    listOfListsOfMatchingCharsInScene = detect_chars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if handle_plate.showSteps == True: # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(len(listOfListsOfMatchingCharsInScene)))    # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if # show steps #########################################################################

    # for each group of matching chars
    # make the possible plate candidate
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        # attempt to extract plate
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)
        # if plate was found
        if possiblePlate.imgPlate is not None:
            # add to list of possible plates
            listOfPossiblePlates.append(possiblePlate)
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")          # 13 with MCLRNF1 image


    for i in range(0, len(listOfPossiblePlates)):   # gethar_dataset plates license
        p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

        cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), handle_plate.SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), handle_plate.SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), handle_plate.SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), handle_plate.SCALAR_RED, 2)

    if handle_plate.showSteps == True: # show steps #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), handle_plate.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), handle_plate.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), handle_plate.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), handle_plate.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    return listOfPossiblePlates
# end function

###################################################################################################
'''
    extract only the list of digit character
'''
def findPossibleCharsInScene(digitEtcModel, imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour

        digit_etc_prediction = predictContour(digitEtcModel, imgThresh, contours[i])

        if handle_plate.showSteps == True: # show steps ###################################################
            cv2.drawContours(imgContours, contours, i, handle_plate.SCALAR_WHITE)
        # end if # show steps #####################################################################

        possibleChar = possible_char.PossibleChar(contours[i])

        # if contour is a possible char, note this does not compare to other chars (yet) . . .
        # add only the digit character
        if digit_etc_prediction.isdigit() and detect_chars.checkIfPossibleChar(possibleChar):
            # increment count of possible chars
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            # and add to list of possible chars
            listOfPossibleChars.append(possibleChar)
        # end if
    # end for

    if handle_plate.showSteps == True: # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))                       # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))       # 131 with MCLRNF1 image
        cv2.imshow("2a", imgContours)
    # end if # show steps #########################################################################

    return listOfPossibleChars
# end function

###################################################################################################
'''
    predict the contour is digit or not
'''
def predictContour(digitEtcModel, imgThresh, contour):
    # contour의 좌료를 가져온다.
    x, y, w, h = cv2.boundingRect(contour)

    # 문자 vector를 가져온다.
    char = imgThresh[y:y + h, x:x + w]

    # preprocess the character and describe it
    char = LicensePlateDetector.preprocessChar(char)

    features = desc.describe(char).reshape(1, -1)
    digitetcprediction = digitEtcModel.predict(features)[0][-1:]

    if handle_plate.showSteps == True: # show steps #######################################################
        print("def predictContour digitetcprediction:[%s]" % (digitetcprediction))
    # end if # show steps #########################################################################

    return digitetcprediction
# end function

###################################################################################################
'''
    make the plate candidates validation
'''
def extractPlate(imgOriginal, listOfMatchingChars):
    # this will be the return value
    possiblePlate = possible_plate.PossiblePlate()

    # sort chars from left to right based on x position
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    # 번호판폭 계산 (the end of the last - the start of the first)
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = detect_chars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # final steps are to perform the actual rotation
    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    # unpack original image width and height
    height, width, numChannels = imgOriginal.shape

    # rotate the entire image
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.imgPlate = imgCropped

    return possiblePlate
# end function












