# USAGE
# python gather_examples.py --images ../full_lp_dataset --examples output/examples
# python gather_examples.py --images input_dataset --examples output_digit/dataset/nochar
# python gather_digitetc.py --i input_dataset --e output_digit/dataset/nochar
# python gather_digitetc.py --i input_dataset --e output/digitetc/e
# python gather_digitetc.py --i input_dataset --e output/examples


# import the necessary packages
from __future__ import print_function
from license_plate_lib.license_plate import LicensePlateDetector
from license_plate_lib.descriptors import BlockBinaryPixelSum
from imutils import paths
import traceback
import argparse
import imutils
import numpy as np
import random
import cv2
import os
from skimage.filters import threshold_local
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the images to be classified")
ap.add_argument("-e", "--examples", required=True, help="path to the output examples directory")
args = vars(ap.parse_args())

# randomly select a portion of the images and initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:int(len(imagePaths) * 0.5)]
counts = {}

# load the character and digit classifiers
allcharModel = pickle.loads(open("output/adv_allchar.cpickle", "br+").read())
charModel = pickle.loads(open("output/adv_char.cpickle", "br+").read())
digitModel = pickle.loads(open("output/adv_digit.cpickle", "br+").read())

# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)


# jeong 추가
dirPath = 'output/etcfiles'




# loop over the images
for imagePath in imagePaths:
	# show the image path
	print("[EXAMINING] {}".format(imagePath))

	try:
		# load the image
		image_org = cv2.imread(imagePath)
		image = image_org.copy()

		# if the width is greater than 640 pixels, then resize the image
		if image.shape[1] > 640:
			image = imutils.resize(image, width=640)

		# initialize the license plate detector and detect characters on the license plate
		lpd = LicensePlateDetector(image, numChars=7)
		plates = lpd.detect()

		# #########################

		V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
		T = threshold_local(V, 19, offset=15, method="gaussian")
		thresh = (V > T).astype("uint8") * 255
		thresh = cv2.bitwise_not(thresh)

		thresh2, cnts, hierarchy0 = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		hierarchy = hierarchy0[0]

		for i in range(len(cnts)):
			cnt = cnts[i]
			x, y, w, h = cv2.boundingRect(cnt)
			rect_area = w * h
			aspect_ratio = float(w) / h

			if (h < 50 and h > 5) and (w < 50 and w > 3):

				char = thresh[y:y + h, x:x + w]

				# preprocess the character and describe it
				char = LicensePlateDetector.preprocessChar(char)
				if char is None:
					continue
				features = desc.describe(char).reshape(1, -1)

				# model에서 결과를 가져온다.
				prediction = digitModel.predict(features)[0][-1:]
				digitprediction = allcharModel.predict(features)[0][-1:]

				# 숫자는 최소 10 pixel 기준
				if digitprediction.isdigit():

					# display the character and wait for a keypress
					cv2.imshow("Char", char)
					key = cv2.waitKey(0)
					cv2.destroyAllWindows()

					# if the '`' key was pressed, then ignore the character
					#if key == ord("`"):
					if key == 27:
						print("[IGNORING] {}".format(imagePath))
						
						# etc character 에 저장
						file_list = sorted([f for f in os.listdir(dirPath)])

						if len(file_list) > 0:
							file_name = file_list[-1][:-4]

						number_files = int(file_name)
						number_index = number_files + 1

						path = "{}/{}.png".format(dirPath, str(number_index).zfill(5))
						cv2.imwrite(path, char)
						continue

					# grab the key that was pressed and construct the path to the output
					# directory
					key = chr(key).upper()
					charDirPath = "{}/{}".format(args["examples"], key)

					# if the output directory does not exist, create it
					if not os.path.exists(charDirPath):
						os.makedirs(charDirPath)

					# write the labeled character to file
					number_files = len([f for f in os.listdir(charDirPath)])
					##count = counts.get(key, 1)

					path = "{}/{}.png".format(charDirPath, str(number_files + 1).zfill(5))
					cv2.imwrite(path, char)

					# 파란색
					cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
					# 부모 contour 목록 구성
					##parentindex.append(hierarchy[i][3])

					print("===================")
					print("prediction [%s]:" % (prediction))
					print("digitprediction [%s]:" % (digitprediction))
					print("hierarchy [%s]" % (hierarchy[i]))
					print("parent index:[%s]" % (hierarchy[i][3]))

				# else:
				# 	# number_files = len([f for f in os.listdir(dirPath)])
				# 	# number_index = number_files + 1
                 #    #
				# 	# path = "{}/{}.png".format(dirPath, str(number_index).zfill(5))
				# 	# cv2.imwrite(path, char)
                #
				# 	# etc character 에 저장
				# 	file_list = sorted([f for f in os.listdir(dirPath)])
                #
				# 	if len(file_list) > 0:
				# 		file_name = file_list[-1][:-4]
                #
				# 	number_files = int(file_name)
				# 	number_index = number_files + 1
                #
				# 	path = "{}/{}.png".format(dirPath, str(number_index).zfill(5))
				# 	cv2.imwrite(path, char)

	# we are trying to control-c out of the script, so break from the loop
	except KeyboardInterrupt:
		break

	# an unknwon error occured for this particular image, so do not process it and display
	# a traceback for debugging purposes
	except:
		print(traceback.format_exc())
		print("[ERROR] {}".format(imagePath))