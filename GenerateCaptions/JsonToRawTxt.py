import time
import os
import sys
import json
import warnings

CAPTION_JSON_FILE = "/.../captions_val2017.json"
OUTPUT_FILE_PATH = "/.../"
CAPTION_TXT_FILE = "caption"
NUM_CAPTION_PER_FILE = 18000

def main():
	counter = 1
	i = 1

	with open(CAPTION_JSON_FILE) as json_file:
		captionFile = json.load(json_file)
		f = open(OUTPUT_FILE_PATH + CAPTION_TXT_FILE + "_" + str(i) + ".txt", "w")

		for p in captionFile['annotations']:
			f.write(p['caption'].replace("\n",""))
			counter+=1
			
			if (counter > len(captionFile['annotations'])):
				f.close()
				break

			if (counter % NUM_CAPTION_PER_FILE  == 0):
				f.close()
				i+=1
				f = open(OUTPUT_FILE_PATH + CAPTION_TXT_FILE + "_" + str(i) + ".txt", "w")
			else:
				f.write("\n")

if __name__ == '__main__':
	warnings.simplefilter(action='ignore', category=FutureWarning)
	main()


