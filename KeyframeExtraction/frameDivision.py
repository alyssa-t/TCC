import cv2
import os
import shutil
import math

VIDEOS_DIR = '/home/alyssa/Videos/'                        # The place to put the video
TARGET_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/' 
TARGET_CANNY_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/Canny/'
VIDEO_FILE_NAME = 'tensao.mp4'
FRAME_PER_SECONDS = 1
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200


class Image_Clustering:
    def __init__(self, video_file=VIDEO_FILE_NAME, image_file_temp='img_%s.png', input_video=True):
        self.video_file = video_file            # Input video file name
        self.image_file_temp = image_file_temp  # Image file name template
        self.input_video = input_video          # If input data is a video

    def video_2_frames(self):
        print('Video to frames...')
        print(VIDEOS_DIR+self.video_file)

        cap = cv2.VideoCapture(VIDEOS_DIR+self.video_file)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
            exit


        print("Opening video stream or file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print ("FPS = "+str(fps))

        # Remove and make a directory
        if os.path.exists(TARGET_IMAGES_DIR):
          shutil.rmtree(TARGET_IMAGES_DIR)  # Delete an entire directory tree
        if not os.path.exists(TARGET_IMAGES_DIR):
            os.makedirs(TARGET_IMAGES_DIR)  # Make a directory
            #os.makedirs(TARGET_CANNY_IMAGES_DIR)

        i = 0
        count = 0

        while(cap.isOpened()):
            flag, frame = cap.read()  # Capture frame-by-frame
            if flag == False:
                break  # A frame is not left
            if (count%(math.ceil(fps/FRAME_PER_SECONDS))== 0):
                cv2.imwrite(TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6), frame)  # Save a frame
              #  frame = cv2.Canny(frame, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
              #  cv2.imwrite(TARGET_CANNY_IMAGES_DIR+self.image_file_temp % str(i).zfill(6), frame)  # Save a frame

            i += 1
            count+=1
        print('Saved '+str(i)+" images")
        cap.release()  # When everything done, release the capture
        print('')

def main(self):
    if self.input_video == True:
        self.video_2_frames()



obj = Image_Clustering()
main(obj)