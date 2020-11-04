import cv2
import os
import shutil
import math
import numpy as np
from keras.preprocessing import image
from progressbar import ProgressBar

class Data2Frames:
	def __init__(self):
		super(Data2Frames, self).__init__()

	def extractFrames(self, VIDEO_FILE_NAME, FRAME_PER_SECONDS, RESIZED_IMAGE_SIZE):
		self.video_file = VIDEO_FILE_NAME            # Input video file name

		print('Trying to read video file from :' + str(self.video_file))
		cap = cv2.VideoCapture(self.video_file)

		if (cap.isOpened()== False):
			print("Error opening video stream or file")
			exit()

		print("Opening file..")
		fps = cap.get(cv2.CAP_PROP_FPS)
		videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		count = 0
		numFrames = 0
		pb = ProgressBar(videoLength).start()
		first = True
		while(cap.isOpened()):
			flag, frame = cap.read()  # Capture frame-by-frame
			if flag == False:
				break  # A frame is not left

			if (count%(math.ceil(fps/FRAME_PER_SECONDS))== 0):
				originalAuxArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				auxArray = cv2.resize(originalAuxArray, (RESIZED_IMAGE_SIZE,RESIZED_IMAGE_SIZE))
				originalAuxArray = np.expand_dims(originalAuxArray, axis=0)
				auxArray = np.expand_dims(auxArray, axis=0)
				if first:
					first = False
					framesArray_resized = auxArray
					framesArray = originalAuxArray
				else:
					framesArray_resized = np.concatenate((framesArray_resized  , auxArray))
					framesArray = np.concatenate((framesArray, originalAuxArray))
				numFrames+=1
				
			pb.update(count)
			count+=1
		print('Read '+str(numFrames)+" images")
		cap.release()  # When everything done, release the capture
		return (framesArray_resized, framesArray )
		
