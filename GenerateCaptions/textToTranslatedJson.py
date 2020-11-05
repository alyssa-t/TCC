import time
import os
import sys
import json
import warnings

ORIGINAL_JSON_FILE = "/.../captions_val2017.json"
TRANSLATED_TXT_FILE_PATH = "/.../"
CAPTION_TXT_FILE = "caption"
TRANSLATED_JSON_FILE = '/.../captions_val2017_translated.json'
NUM_CAPTION_PER_FILE = 18000

def main():
	counter = 1
	i = 1

	with open(ORIGINAL_JSON_FILE) as json_file:
		originalcaptionFile = json.load(json_file)
		f = open(TRANSLATED_TXT_FILE_PATH + CAPTION_TXT_FILE + "_" + str(i) + ".txt", "r")
		for p in originalcaptionFile['annotations']:
			auxCaption = f.readline().replace("\n","")
			p['caption'] = auxCaption
			counter+=1
			
			if (counter > len(originalcaptionFile['annotations'])):
				f.close()
				break

			if (counter % NUM_CAPTION_PER_FILE  == 0):
				f.close()
				i+=1
				f = open(TRANSLATED_TXT_FILE_PATH + CAPTION_TXT_FILE + "_" + str(i) + ".txt", "r")

	with open(TRANSLATED_JSON_FILE, 'w') as outfile:
		json.dump(originalcaptionFile, outfile)


if __name__ == '__main__':
	warnings.simplefilter(action='ignore', category=FutureWarning)
	main()


