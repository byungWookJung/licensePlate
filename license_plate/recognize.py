#-*- coding: utf-8 -*-
# USAGE
# python recognize.py
# -i ../hyundai_car_data/sin -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle
# -i ../error_dataset -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle

# import the necessary packages
from __future__ import print_function
from license_plate_lib.license_plate import LicensePlateDetector
from license_plate_lib.license_plate_class import ANPRChar
from license_plate_lib.descriptors import BlockBinaryPixelSum
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import cv2
from skimage.filters import threshold_local
import collections
import operator
from license_plate_lib import detect_plates
from license_plate_lib import detect_chars
from license_plate_lib import handle_plate
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the images to be classified")
ap.add_argument("-de", "--digitetc-classifier", required=True,
	help="path to the output all character classifier")
ap.add_argument("-k", "--hangul-classifier", required=True,
	help="path to the output kor character classifier")
ap.add_argument("-c", "--char-classifier", required=True,
	help="path to the output character classifier")
ap.add_argument("-d", "--digit-classifier", required=True,
	help="path to the output digit classifier")
args = vars(ap.parse_args())

# load the character and digit classifiers
digitetcModel = pickle.loads(open(args["digitetc_classifier"], "br+").read())
hangulModel = pickle.loads(open(args["hangul_classifier"], "br+").read())
charModel = pickle.loads(open(args["char_classifier"], "br+").read())
digitModel = pickle.loads(open(args["digit_classifier"], "br+").read())

# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# license plate variable
licPlate = None

##########################################################################################
'''
	name   : predict_license_plate
	detail : predict license contents
'''
def predict_license_plate(imagePath, charModel, hangulModel, digitModel, digitetcModel):

	# load the image
	print(imagePath[imagePath.rfind("/") + 1:])
	image = cv2.imread(imagePath).copy()
	imgOriginalScene = cv2.imread(imagePath)
	imgOriginalScene = handle_plate.makeHistogramEqualization(imgOriginalScene)

	# if the width is greater than 640 pixels, then resize the image
	if image.shape[1] > 640:
		image = imutils.resize(image, width=640)
		imgOriginalScene = imutils.resize(image, width=640)

	if imgOriginalScene is None:  # if image was not read successfully
		print("\nerror: image not read from file \n\n")  # print error message to std out
		os.system("pause")  # pause so user can see error message
		#return  # and exit program
	# end if

	listOfPossiblePlates = detect_plates.detectPlatesInScene(digitetcModel, image)           # detect plates

	listOfPossiblePlates = detect_chars.detectCharsInPlates(digitetcModel, hangulModel, charModel, digitModel, listOfPossiblePlates)  # detect chars in plates

	cv2.imshow("imgOriginalScene", imgOriginalScene)  # show scene image

	if len(listOfPossiblePlates) == 0:  # if no plates were found
		print("\nno license plates were detected\n")  # inform user no plates were found
	else:  # else
		# if we get in here list of possible plates has at leat one plate

		# sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
		# listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
		listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.numDigits), reverse=True)

		# suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
		licPlate = listOfPossiblePlates[0]
		# TODO : 삭제
		for testPlate in listOfPossiblePlates:
			handle_plate.drawRedRectangleAroundPlate2(imgOriginalScene, testPlate)
		# end of TODO : 삭제

		cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
		cv2.imshow("imgThresh", licPlate.imgThresh)

		if len(licPlate.strChars) == 0:  # if no chars were found in the plate
			print("\nno characters were detected\n\n")  # show message
			#return  # and exit program
		# end if

		handle_plate.drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate

		print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out

		handle_plate.writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # write license plate text on the image

		cv2.imshow("imgOriginalScene", imgOriginalScene)  # re-show scene image

		cv2.imwrite("imgOriginalScene.png", imgOriginalScene)  # write image out to file
	# end if else

	# return the prediction
	return licPlate
##########################################################################################

##########################################################################################
# loop over the images
for imagePath in sorted(list(paths.list_images(args["images"]))):
	# load the image
	print(imagePath[imagePath.rfind("/") + 1:])
	image = cv2.imread(imagePath).copy()
	imgOriginalScene = cv2.imread(imagePath)
	imgOriginalScene = handle_plate.makeHistogramEqualization(imgOriginalScene)

	# if the width is greater than 640 pixels, then resize the image
	if image.shape[1] > 640:
		image = imutils.resize(image, width=640)
		imgOriginalScene = imutils.resize(image, width=640)

	##########################################################################################

	if imgOriginalScene is None:  # if image was not read successfully
		print("\nerror: image not read from file \n\n")  # print error message to std out
		os.system("pause")  # pause so user can see error message
		#return  # and exit program
	# end if

	listOfPossiblePlates = detect_plates.detectPlatesInScene(digitetcModel, image)           # detect plates

	listOfPossiblePlates = detect_chars.detectCharsInPlates(digitetcModel, hangulModel, charModel, digitModel, listOfPossiblePlates)  # detect chars in plates

	cv2.imshow("imgOriginalScene", imgOriginalScene)  # show scene image

	if len(listOfPossiblePlates) == 0:  # if no plates were found
		print("\nno license plates were detected\n")  # inform user no plates were found
	else:  # else
		# if we get in here list of possible plates has at leat one plate

		# sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
		listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

		# suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
		licPlate = listOfPossiblePlates[0]
		# TODO : 삭제
		for testPlate in listOfPossiblePlates:
			handle_plate.drawRedRectangleAroundPlate2(imgOriginalScene, testPlate)
		# end of TODO : 삭제

		cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
		cv2.imshow("imgThresh", licPlate.imgThresh)

		if len(licPlate.strChars) == 0:  # if no chars were found in the plate
			print("\nno characters were detected\n\n")  # show message
			#return  # and exit program
		# end if

		handle_plate.drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate

		print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out

		handle_plate.writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # write license plate text on the image

		cv2.imshow("imgOriginalScene", imgOriginalScene)  # re-show scene image

		cv2.imwrite("imgOriginalScene.png", imgOriginalScene)  # write image out to file
	# end if else

	# display the output image
	# cv2.imshow("image", image)
	# cv2.waitKey(0)
	
# end of for

##########################################################################################
