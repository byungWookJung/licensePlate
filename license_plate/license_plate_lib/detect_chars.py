#-*- coding: utf-8 -*-
# DetectChars.py

import cv2
import numpy as np
import math
import random

from license_plate_lib import handle_plate
from license_plate_lib import preprocess
from license_plate_lib import possible_char
import os
from license_plate_lib.descriptors import BlockBinaryPixelSum
from license_plate_lib.license_plate import LicensePlateDetector
import imutils

# module level variables ##########################################################################

kNearest = cv2.ml.KNearest_create()

# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
#MIN_PIXEL_WIDTH = 2
MIN_PIXEL_WIDTH = 1
MIN_PIXEL_HEIGHT = 8

#MIN_ASPECT_RATIO = 0.25
MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 1.0

#MIN_PIXEL_AREA = 80
MIN_PIXEL_AREA = 50

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
#MAX_CHANGE_IN_HEIGHT = 0.2
MAX_CHANGE_IN_HEIGHT = 0.3

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

kor_chars_dic = {"ga":"가", "geo":"거", "go":"고", "gu":"구", "na":"나", "neo":"너", "no":"노", "nu":"누", "da":"다", "deo":"더", "do":"도", "du":"두", "la":"라", "leo":"러", "lo":"로", "lu":"루", "ma":"마", "meo":"머", "mo":"모", "mu":"무"
, "ba":"바", "beo":"버", "bo":"보", "bu":"부", "sa":"사", "seo":"서", "so":"소", "su":"수", "aa":"아", "eo":"어", "oo":"오", "uu":"우", "ja":"자", "jeo":"저", "jo":"조", "ju":"주", "ha":"하", "heo":"허", "ho":"호"}

###################################################################################################
def detectCharsInPlates(digitEtcModel, hangulModel, charModel, digitModel, listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # if list of possible plates is empty
        return listOfPossiblePlates             # return
    # end if

    '''at this point we can be sure the list of possible plates has at least one plate'''
    # for each possible plate, this is a big for loop that takes up most of the function
    for possiblePlate in listOfPossiblePlates:

        # TODO : 확인중 (배율 1.5 최상인지 확인)
        # thresh 하기전 이미지 확대 1.5
        possiblePlate.imgPlate = cv2.resize(possiblePlate.imgPlate, (0, 0), fx=1.5, fy=1.5)

        # preprocess to get grayscale and threshold images
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess.preprocess(possiblePlate.imgPlate)

        if handle_plate.showSteps == True: # show steps ###################################################
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # TODO : Delete
        # # increase size of plate image for easier viewing and char detection
        # possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if handle_plate.showSteps == True: # show steps ###################################################
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if # show steps #####################################################################

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if handle_plate.showSteps == True: # show steps ###################################################
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # clear the contours list

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, handle_plate.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if # show steps #####################################################################

        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if handle_plate.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if # show steps #####################################################################

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# if no groups of matching chars were found in the plate

            if handle_plate.showSteps == True: # show steps ###############################################
                print("chars found in plate number " + str(intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # show steps #################################################################

            possiblePlate.strChars = ""
            continue						# go back to top of for loop
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # within each list of matching chars
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # and remove inner overlapping chars
        # end for

        if handle_plate.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if # show steps #####################################################################

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if handle_plate.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, handle_plate.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if # show steps #####################################################################

        possiblePlate.strChars, possiblePlate.numDigits = recognizeCharsInPlate(digitEtcModel, hangulModel, charModel, digitModel, possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if handle_plate.showSteps == True: # show steps ###################################################
            print("chars found in plate number " + str(intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # show steps #####################################################################

    # end of big for loop that takes up most of the function

    if handle_plate.showSteps == True:
        print("char detection complete, click on any image and press a key to continue . . .")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # for each contour
    for contour in contours:
        possibleChar = possible_char.PossibleChar(contour)

        # if contour is a possible char, note this does not compare to other chars (yet) . . .
        if checkIfPossibleChar(possibleChar):
            # add to list of possible chars
            listOfPossibleChars.append(possibleChar)
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
'''
    this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    note that we are not (yet) comparing the char to other chars to look for a group
'''
def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
'''
    with this function, we start off with all the possible chars in one big list
    the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    note that chars that are not found to be in a group of matches do not need to be considered further
'''
def findListOfListsOfMatchingChars(listOfPossibleChars):

    # this will be the return value
    listOfListsOfMatchingChars = []

    # for each possible char in the one big list of chars
    for possibleChar in listOfPossibleChars:

        # find all chars in the big list that match the current char
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)

        # also add the current char to current possible list of matching chars
        listOfMatchingChars.append(possibleChar)

        # if current possible list of matching chars is not long enough to constitute a possible plate
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            # jump back to the top of the for loop and try again with next char, note that it's not necessary
            # to save the list in any way since it did not have enough chars to be a possible plate
            continue
        # end if

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        # so add to our list of lists of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        # recursive call
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)

        # for each list of matching chars found by recursive call
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            # add to our original list of lists of matching chars
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        # end for

        # exit for
        break

    # end for

    return listOfListsOfMatchingChars
# end function

'''
    the purpose of this function is, given a possible char and a big list of possible chars,
    find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
'''
def findListOfMatchingChars(possibleChar, listOfChars):

    # this will be the return value
    listOfMatchingChars = []

    # for each char in big list
    for possibleMatchingChar in listOfChars:

        # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
        if possibleMatchingChar == possibleChar:
            # then we should not include it in the list of matches b/c that would end up double including the current char
            # so do not add to list of matches and jump back to top of for loop
            continue
        # end if

        # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            # if the chars are a match, add the current char to list of matching chars
            listOfMatchingChars.append(possibleMatchingChar)
        # end if
    # end for

    return listOfMatchingChars                  # return result
# end function

###################################################################################################
'''
    use Pythagorean theorem to calculate distance between two chars
'''
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
'''
    use basic trigonometry (SOH CAH TOA) to calculate angle between chars
'''
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
    if fltAdj != 0.0:
        # if adjacent is not zero, calculate angle
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
        fltAngleInRad = 1.5708
    # end if

    # calculate angle in degrees
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)

    return fltAngleInDeg
# end function

###################################################################################################
'''
    if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
    this is to prevent including the same char twice if two contours are found for the same char,
    for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
'''
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # if current char and other char are not the same char . . .
                                                                            # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # if we get in here we have found overlapping chars
                                # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # then remove current char
                        # end if
                    else:                                                                       # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################
'''
    this is where we apply the actual char recognition
'''
def recognizeCharsInPlate(digitEtcModel, hangulModel, charModel, digitModel, imgThresh, listOfMatchingChars):
    strChars = ""               # this will be the return value, the chars in the lic plate

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # make color version of threshold image so we can draw contours in color on it

    # cv2.imshow("imgThresh", imgThresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # korean character
    kor_char = None
    kor_prediction = ""

    # digit count
    digit_count = 0

    #for currentChar in listOfMatchingChars:                                         # for each char in plate
    for (i, currentChar) in enumerate(listOfMatchingChars):                                         # for each char in plate

        #####################################################
        # cnt = contours[i]
        x, y, w, h = cv2.boundingRect(currentChar.contour)
        if i == 0:
            temp_x = x
        # end if
        rect_area = w * h
        aspect_ratio = float(w) / float(h)
        # 문자 vector를 가져온다.
        char = imgThresh[y:y + h, x:x + w]

        # preprocess the character and describe it
        char = LicensePlateDetector.preprocessChar(char)
        if char is None:
            continue
        features = desc.describe(char).reshape(1, -1)
        prediction = digitModel.predict(features)[0][-1:]
        digitprediction = digitEtcModel.predict(features)[0][-1:]

        print("prediction:[%s]"  % (prediction))
        print("digitprediction:[%s]"  % (digitprediction))
        # continue if char is not digit
        if digitprediction.isdigit() and prediction == digitprediction:
            digit_count += 1
        else:
            continue

        # get korean character
        # save korean character as a file
        if i == 1:
            result_w = abs(temp_x - x)

            # 한글 문자 vector를 가져온다.
            # kor_char = imgThresh[y:y+h, int((x + w * 1.1)):int((x + w * 1.5) + w)]
            # kor_char = imgThresh[y:y + h, int((x + h * 0.5)):int((x + h * 1.1))]
            # kor_char = imgThresh[y:y + h, int((x + result_w * 0.9)):int((x + result_w) + result_w * 1.1)]
            kor_char = imgThresh[y:y + h, int((x + result_w - 2)):int((x + result_w) + result_w + 1)]
            cv2.imshow("kor1", kor_char)
            # get korean character thresh
            kor_char = getKorCharByContour(kor_char)
            cv2.imshow("kor2", kor_char)
            # 한글 파일 저장
            saveKorCharFile(kor_char)

        # end if

        strChars += prediction

    # end of for

    '''korean character predict'''
    # preprocess the character and describe it
    if kor_char is not None:
        ''' 한글 contour 문제로 삭제
        kor_char = LicensePlateDetector.preprocessChar(kor_char)'''
        features = desc.describe(kor_char).reshape(1, -1)
        kor_prediction = hangulModel.predict(features)[0]
        # cv2.waitKey(0)
        tempChars = ''
        if len(strChars) >= 6:
            if kor_chars_dic.get(kor_prediction):
                kor_prediction = kor_chars_dic[kor_prediction]
            # end if
            tempChars = strChars[:2] + kor_prediction + strChars[-4:]
            strChars = tempChars
        # end if

        print("=======================================================")
        print("kor_prediction:[%s]" % (kor_prediction))
        print("=======================================================")

    # end process of kor character

    if handle_plate.showSteps == True: # show steps #######################################################
        cv2.imshow("10", imgThreshColor)
    # end if # show steps #########################################################################

    return strChars, digit_count
# end function

###################################################################################################
'''
    get the korean character thresh by contour
'''
def getKorCharByContour(kor_char):

    kor_char_cont, contours, hierarchy0 = cv2.findContours(kor_char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        cont = contours[0]
        x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(cont)
        kor_char = kor_char_cont[y_temp:y_temp + h_temp, x_temp:x_temp + w_temp]
    # end if

    if len(contours) == 2:
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for k in range(len(contours)):
            cont = contours[k]
            x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(cont)

            if k == 0:
                # x_min, y_min, x_max, y_max
                x_min = x_temp
                x_max = x_temp + w_temp
                y_min = y_temp
                y_max = y_temp + h_temp

            #elif k == 1:
            else:

                # get x_min, y_min
                if x_temp < x_min:
                    x_min = x_temp
                if y_temp < y_min:
                    y_min = y_temp

                # get x_max, y_max
                if x_temp + w_temp > x_max:
                    x_max = x_temp + w_temp
                if y_temp + h_temp > y_max:
                    y_max = y_temp + h_temp
                # end if

        # end for
        kor_char = kor_char_cont[y_min:y_max, x_min:x_max]

    # end if

    return kor_char
# end function

###################################################################################################
'''
    save the korean character as a single file
'''
def saveKorCharFile(kor_char):

    # 한글 파일 저장
    korCharDirPath = "./hangul_temp"
    if not os.path.exists(korCharDirPath):
        os.makedirs(korCharDirPath)
    # end if

    file_list = sorted([f for f in os.listdir(korCharDirPath)])

    file_name = '0'
    if len(file_list) > 0:
        file_name = file_list[-1][:-4]
    # end if

    number_files = int(file_name)
    number_index = number_files + 1

    pngPath = "{}/{}.png".format(korCharDirPath, str(number_index).zfill(6))
    cv2.imwrite(pngPath, kor_char)
    # end 한글 파일 저장

# end function

###################################################################################################
'''
    add margin to the char of 2 channel image
'''
def addMarginXOf2Channel(char):

    margin = 2

    y, x = char.shape
    img_temp = np.zeros((y, x + margin * 2), np.uint8)
    img_temp[0:y, margin:x + margin] = char

    return img_temp
# end function





