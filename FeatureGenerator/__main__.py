import tensorflow as tf
import os
import warnings
from .featureGenerator import *

YOLO_RESULT_FILE = "/.../result.txt"
PATH_TMP_FOLDER = "/.../tmp/"
PATH_NPY_FILES = "/.../trainImages/npy/"
BATCH_SIZE_INCEPTIONV3 = 16
NTH_SLASH = 6

def main():	
	#Code for setup GPU
	#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	gpu_devices = tf.config.experimental.list_physical_devices('GPU')
	for device in gpu_devices:
		tf.config.experimental.set_memory_growth(device, True)
	tf.compat.v1.enable_eager_execution()

	try:
		os.stat(PATH_TMP_FOLDER)
	except:
		os.mkdir(PATH_TMP_FOLDER)  
	try:
		os.stat(PATH_NPY_FILES)
	except:
		os.mkdir(PATH_NPY_FILES)
	
	featureGen = Feature_Generator(YOLO_RESULT_FILE, PATH_TMP_FOLDER, PATH_NPY_FILES, BATCH_SIZE_INCEPTIONV3, NTH_SLASH)
	featureGen.cropImage()
	featureGen.extractFeature()
	shutil.rmtree(PATH_TMP_FOLDER)

if __name__ == '__main__':
	warnings.simplefilter(action='ignore', category=FutureWarning)
	main()
