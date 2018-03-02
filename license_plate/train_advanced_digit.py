#-*- coding: utf-8 -*-

# USAGE
# python train_advanced_digit.py -s output -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle

# import the necessary packages
from __future__ import print_function
from license_plate_lib.license_plate import LicensePlateDetector
from license_plate_lib.descriptors import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import pickle
import random
import glob
import cv2

from license_plate_lib import detect_chars

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="path to the training samples directory")
ap.add_argument("-de", "--digitetc-classifier", required=True, help="path to the output all character classifier")
ap.add_argument("-k", "--hangul-classifier", required=True, help="path to the output character classifier")
ap.add_argument("-c", "--char-classifier", required=True, help="path to the output character classifier")
ap.add_argument("-d", "--digit-classifier", required=True, help="path to the output digit classifier")
'''ap.add_argument("-m", "--min-samples", type=int, default=20, help="minimum # of samples per character")'''
args = vars(ap.parse_args())

# initialize the descriptor
blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

# initialize the data and labels for the alphabet and digits
digitEtcData = []
digitEtcLabels = []

hangulData = []
hangulLabels = []

alphabetData = []
alphabetLabels = []

digitsData = []
digitsLabels = []

# define cpickle file path
alpha_path = args["samples"] + "/dataset/alpha"
hangul_path = args["samples"] + "/dataset/hangul"
digit_path = args["samples"] + "/dataset/digit"
digitetc_path = args["samples"] + "/dataset/digitetc"

# loop over the alphabet character paths
for alphabetPath in sorted(glob.glob(alpha_path + "/*")):
	# replace string '\\' to '/'
	alphabetPath = alphabetPath.replace('\\', '/')
	# extract the sample name, grab all images in the sample path, and sample them
	alphaName = alphabetPath[alphabetPath.rfind("/") + 1:]
	imagePaths = list(paths.list_images(alphabetPath))
	'''imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))'''

	# loop over all images in the digitetc path
	for imagePath in imagePaths:
		# load the character, convert it to grayscale, preprocess it, and describe it
		char = cv2.imread(imagePath)
		if char is None:
			continue
		# end if
		char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
		char = LicensePlateDetector.preprocessChar(char)
		features = desc.describe(char)

		alphabetData.append(features)
		alphabetLabels.append(alphaName)
	# end for
# end for

# loop over the hangul character paths
for hangulPath in sorted(glob.glob(hangul_path + "/*")):
	# replace string '\\' to '/'
	hangulPath = hangulPath.replace('\\', '/')
	# extract the sample name, grab all images in the sample path, and sample them
	hangulName = hangulPath[hangulPath.rfind("/") + 1:]
	imagePaths = list(paths.list_images(hangulPath))
	'''imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))'''

	# loop over all images in the digitetc path
	for imagePath in imagePaths:
		# load the character, convert it to grayscale, preprocess it, and describe it
		char = cv2.imread(imagePath)
		if char is None:
			continue
		# end if
		char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)

		char = detect_chars.getKorCharByContour(char)

		''' 한글에서는 contour 가 하나가 아니므로 삭제
		char = LicensePlateDetector.preprocessChar(char)'''
		features = desc.describe(char)

		hangulData.append(features)
		hangulLabels.append(hangulName)
	# end for
# end for

# loop over the digitetc character paths
for digitetcPath in sorted(glob.glob(digitetc_path + "/*")):
	# replace string '\\' to '/'
	digitetcPath = digitetcPath.replace('\\', '/')
	# extract the sample name, grab all images in the sample path, and sample them
	digitEtcName = digitetcPath[digitetcPath.rfind("/") + 1:]
	imagePaths = list(paths.list_images(digitetcPath))
	'''imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))'''

	# loop over all images in the digitetc path
	for imagePath in imagePaths:
		# load the character, convert it to grayscale, preprocess it, and describe it
		char = cv2.imread(imagePath)
		if char is None:
			continue
		# end if
		char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
		char = LicensePlateDetector.preprocessChar(char)
		features = desc.describe(char)

		digitEtcData.append(features)
		digitEtcLabels.append(digitEtcName)
	# end for
# end for

# loop over the digit character paths
for digitPath in sorted(glob.glob(digit_path + "/*")):
	# replace string '\\' to '/'
	digitPath = digitPath.replace('\\', '/')
	# extract the sample name, grab all images in the sample path, and sample them
	####sampleName = samplePath[samplePath.rfind("/") + 1:][-1:]
	digitName = digitPath[digitPath.rfind("/") + 1:]
	imagePaths = list(paths.list_images(digitPath))
	'''imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))'''

	# loop over all images in the sample path
	for imagePath in imagePaths:
		# load the character, convert it to grayscale, preprocess it, and describe it
		char = cv2.imread(imagePath)
		if char is None:
			continue
		# end if
		char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
		char = LicensePlateDetector.preprocessChar(char)
		features = desc.describe(char)

		# check to see if we are examining a digit
		if digitName.isdigit():
			digitsData.append(features)
			digitsLabels.append(digitName)
		# end if
	# end for
# end for

# train the digitetc character classifier
print("[INFO] fitting digitetc character model...")
digitEtcModel = LinearSVC(C=1.0, random_state=42)
digitEtcModel.fit(digitEtcData, digitEtcLabels)

# train the hangul character classifier
print("[INFO] fitting hangul character model...")
hangulModel = LinearSVC(C=1.0, random_state=42)
hangulModel.fit(hangulData, hangulLabels)

# train the alphabet character classifier
print("[INFO] fitting alphabet character model...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(alphabetData, alphabetLabels)

# train the digit character classifier
print("[INFO] fitting digit model...")
digitModel = LinearSVC(C=1.0, random_state=42)
digitModel.fit(digitsData, digitsLabels)

# dump the digitetc character classifier to file
print("[INFO] dumping digitetc character model...")
f = open(args["digitetc_classifier"], "wb")
f.write(pickle.dumps(digitEtcModel))
f.close()

# dump the hangul classifier to file
print("[INFO] dumping hangul character model...")
f = open(args["hangul_classifier"], "wb")
f.write(pickle.dumps(hangulModel))
f.close()

# dump the alphabet character classifier to file
print("[INFO] dumping alphabet character model...")
f = open(args["char_classifier"], "wb")
f.write(pickle.dumps(charModel))
f.close()

# dump the digit classifier to file
print("[INFO] dumping digit character model...")
f = open(args["digit_classifier"], "wb")
f.write(pickle.dumps(digitModel))
f.close()