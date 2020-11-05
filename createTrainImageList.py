import time
import os
import sys
import json
import warnings


TRAIN_IMAGE_FILEPATH = "/.../"
OUTPUT_TXT_FILEPATH = "/.../"
OUTPUT_TXT_FILENAME = "train.txt"
NUM_IMAGE = 50000

def main():

	imagesPath = [f for f in os.listdir(TRAIN_IMAGE_FILEPATH) if f[-4:] in ['.png', '.jpg']]

	with open(OUTPUT_TXT_FILENAME, 'w') as out_file:
		for path in imagesPath:
			input(TRAIN_IMAGE_FILEPATH+ path)
			out_file.write(TRAIN_IMAGE_FILEPATH+ path+"\n")

if __name__ == '__main__':
	warnings.simplefilter(action='ignore', category=FutureWarning)
	main()


