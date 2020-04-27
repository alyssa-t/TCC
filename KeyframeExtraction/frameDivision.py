import cv2
import os
import shutil
import math
from progressbar import ProgressBar

VIDEOS_DIR = '../data/Videos/'                        # The place to put the video
TARGET_IMAGES_DIR = '../frames/'                 # The place to put the resulting images
VIDEO_FILE_NAME = 'video.mp4'                           # Video file name
FRAME_PER_SECONDS = 1


class Data2Frames:
    def __init__(self, video_file=VIDEO_FILE_NAME, image_file_temp='img_%s.png', input_video=True):
        self.video_file = video_file            # Input video file name
        self.image_file_temp = image_file_temp  # Image file name template
        self.input_video = input_video          # If input data is a video

    def main(self):
        if self.input_video == True:
            self.video_2_frames()

    def video_2_frames(self):
        print('Video to frames...')
        print(VIDEOS_DIR+self.video_file)
        cap = cv2.VideoCapture(VIDEOS_DIR+self.video_file)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")
            exit()

        print("Opening video stream or file..")
        fps = cap.get(cv2.CAP_PROP_FPS)
        videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Remove and make a directory
        if os.path.exists(TARGET_IMAGES_DIR):
          shutil.rmtree(TARGET_IMAGES_DIR)  # Delete an entire directory tree
        if not os.path.exists(TARGET_IMAGES_DIR):
            os.makedirs(TARGET_IMAGES_DIR)  # Make a directory

        count = 0
        pb = ProgressBar(videoLength).start()
        while(cap.isOpened()):
            flag, frame = cap.read()  # Capture frame-by-frame
            if flag == False:
                break  # A frame is not left
            if (count%(math.ceil(fps/FRAME_PER_SECONDS))== 0):
                cv2.imwrite(TARGET_IMAGES_DIR+self.image_file_temp % str(count).zfill(6), frame)  # Save a frame
                pb.update(count)
            count+=1
        print ('')
        print('Saved '+str(count)+" images")
        cap.release()  # When everything done, release the capture




if __name__ == "__main__":
    Data2Frames().main()
