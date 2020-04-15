
import cv2
import os
import shutil

VIDEOS_DIR = '/home/alyssa/Vídeos/'                        # The place to put the video
TARGET_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/' 

        


class Image_Clustering:
    def __init__(self, n_clusters=50, video_file='DouMori.avi', image_file_temp='img_%s.png', input_video=True):
        self.video_file = video_file            # Input video file name
        self.image_file_temp = image_file_temp  # Image file name template
        self.input_video = input_video          # If input data is a video

    def video_2_frames(self):
        print('Video to frames...')
        print(VIDEOS_DIR+self.video_file)
        #cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        cap = cv2.VideoCapture(VIDEOS_DIR+self.video_file)
        # Remove and make a directory
        if os.path.exists(TARGET_IMAGES_DIR):
          shutil.rmtree(TARGET_IMAGES_DIR)  # Delete an entire directory tree
        if not os.path.exists(TARGET_IMAGES_DIR):
            os.makedirs(TARGET_IMAGES_DIR)  # Make a directory

        i = 0
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        while(cap.isOpened()):
            flag, frame = cap.read()  # Capture frame-by-frame
            if flag == False:
                break  # A frame is not left
            cv2.imwrite(TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6), frame)  # Save a frame
            i += 1
            print('Save', TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6))

        cap.release()  # When everything done, release the capture
        print('')

def main(self):
    if self.input_video == True:
        self.video_2_frames()



obj = Image_Clustering()
main(obj)