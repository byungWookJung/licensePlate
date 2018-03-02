# USAGE
# python gather_examples.py
# -i ../hyundai_car_data/sin -e output/examples -d output/adv_digit.cpickle
# -i ../dataset/used_dataset -e output/examples
# -i ../dataset/add_dataset -e output/examples
# -i ../dataset/test_dataset -e output/examples

# import the necessary packages
from __future__ import print_function
from imutils import paths
import traceback
import argparse
import imutils
import numpy as np
import random
import cv2
import os

from license_plate_lib import preprocess
from license_plate_lib import detect_plates
from license_plate_lib import detect_chars
from license_plate_lib import handle_plate
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the images to be classified")
ap.add_argument("-e", "--examples", required=True, help="path to the output examples directory")
ap.add_argument("-d", "--digit-classifier", required=True, help="path to the output digit classifier")
args = vars(ap.parse_args())

digitModel = pickle.loads(open(args["digit_classifier"], "br+").read())

# randomly select a portion of the images and initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["images"]))
imagePaths = imagePaths[:int(len(imagePaths) * 0.5)]
counts = {}

# hangulAutoSave = True
hangulAutoSave = False

def main():
    for imagePath in imagePaths:
        print("[EXAMINING] {}".format(imagePath))

        try:
            image = cv2.imread(imagePath).copy()
            imgOriginalScene = cv2.imread(imagePath)

            if image.shape[1] > 640:
                image = imutils.resize(image, width=640)
                imgOriginalScene = imutils.resize(imgOriginalScene, width=640)

            if imgOriginalScene is None :
                print("\n에러 : image not read from file \n\n")
                os.system("pause")

            listOfPossiblePlates = detect_plates.detectPlatesInScene(digitModel, image)  # detect plates

            chars = getDetectChars(listOfPossiblePlates)  # detect chars in plates

            if len(listOfPossiblePlates) == 0:  # if no plates were found
                print("\n자동차 번호판을 찾지 못했습니다.\n")
            else:
                listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)


            if hangulAutoSave == False:
                for char in chars:
                    # display the character and wait for a keypress
                    char = cv2.resize(char, (0, 0), fx=1.6, fy=1.6)
                    cv2.imshow("image ", image)
                    cv2.imshow("Char", char)
                    key = cv2.waitKey(0)

                    # if the '`' key was pressed, then ignore the character
                    if key == ord("`"):
                        print("[IGNORING] {}".format(imagePath))
                        continue
                    # end if

                    # grab the key that was pressed and construct the path to the output
                    key = chr(key).upper()
                    dirPath = "{}/{}".format(args["examples"], key)

                    if not os.path.exists(dirPath):
                        os.makedirs(dirPath)
                    # end if

                    # write the labeled character to file
                    file_list = sorted([f for f in os.listdir(dirPath)])

                    if len(file_list) > 0:
                        file_name = file_list[-1]
                        s = os.path.splitext(file_name)
                        file_name = s[0]
                    else:
                        file_name = 0
                    # end if

                    number_files = int(file_name)
                    print("마지막 file Number : %s" % number_files)
                    number_index = number_files + 1

                    path = "{}/{}.png".format(dirPath, str(number_index).zfill(6))
                    cv2.imwrite(path, char)
                # end for
            # end if

        except KeyboardInterrupt:
            break

        except:
            print(traceback.format_exc())
            print("[ERROR] {}".format(imagePath))
    # end for


def getDetectChars(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []
    chars = []

    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates
    # end if

    for possiblePlate in listOfPossiblePlates:

        possiblePlate.imgPlate = cv2.resize(possiblePlate.imgPlate, (0, 0), fx=1.6, fy=1.6)
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess.preprocess(possiblePlate.imgPlate)

        if handle_plate.showSteps == True:
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if # show steps

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if handle_plate.showSteps == True: # show steps
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if # show steps

        listOfPossibleCharsInPlate = detect_chars.findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if handle_plate.showSteps == True: # show steps
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # clear the contours list

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, handle_plate.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if # show steps

        listOfListsOfMatchingCharsInPlate = detect_chars.findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if handle_plate.showSteps == True: # show steps
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
        # end if # show steps

        if (len(listOfListsOfMatchingCharsInPlate) == 0):
            if handle_plate.showSteps == True: # show steps
                print("chars found in plate number " + str(intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # show steps

            possiblePlate.strChars = ""
            continue
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)
            listOfListsOfMatchingCharsInPlate[i] = detect_chars.removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])
        # end for

        if handle_plate.showSteps == True: # show steps
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
        # end if # show steps

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

        if handle_plate.showSteps == True: # show steps
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, handle_plate.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if # show steps

        chars = getChars(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate, chars)

    # end of big for loop that takes up most of the function

    if handle_plate.showSteps == True:
        print("char detection complete, click on any image and press a key to continue . . .")
        cv2.waitKey(0)
    # end if

    return chars
# end function

def getChars(imgThresh, listOfMatchingChars, chars):

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)

    for i, currentChar in enumerate(listOfMatchingChars):
        c_x = currentChar.intBoundingRectX
        c_y = currentChar.intBoundingRectY
        c_w = currentChar.intBoundingRectWidth
        c_h = currentChar.intBoundingRectHeight

        char = imgThreshColor.__copy__()[c_y:c_y + c_h, c_x:c_x + c_w]

        chars.append(char)
        if i == 0:
            tmp_c_w = c_w
        elif i == 1:
            two_c_w = c_w
            if c_w < tmp_c_w:
                c_w = tmp_c_w
            # 한글 문자 vector를 가져온다.
            kor_char = imgThresh[c_y:c_y + c_h, int((c_x + two_c_w * 1.1)):int((c_x + two_c_w * 1.5) + c_w)]
            kor_char = detect_chars.getKorCharByContour(kor_char)

            if hangulAutoSave == True:
                # 한글 파일 저장
                detect_chars.saveKorCharFile(kor_char)

            chars.append(kor_char)

        # cv2.rectangle(imgThreshColor, (c_x, c_y), ((c_x + c_w), (c_y + c_h)), Main.SCALAR_GREEN, 2)
    # end for

    if handle_plate.showSteps == True: # show steps
        cv2.imshow("10", imgThreshColor)
    # end if # show steps

    return chars
# end function

# start
main()