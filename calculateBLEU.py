
import json
import numpy as np

# initialize vectors
ids_val = []
with open("new_ids_val.txt", 'r') as text:
	for line in text:
		ids_val.append(line.split('\n')[0])

print(ids_val)
input("enter")

lines_result=[]
lines_right=[]
temp=[]
found = False
"""
with open ("result_sorted.txt", 'w') as outext:
	for ids in ids_val:
		with open("result.out", 'r') as text:
			for line in text:
				if (found) and (line[0] == "0"):
					found = False
					continue
				if (not found) and (line.split("\n")[0]==ids):
					outext.writelines(ids+"\n")
					found = True
					continue
				if (found):
					outext.writelines(line)
"""

with open("right_annots_val.txt", 'r') as text:
	for line in text:
		lines_right.append(line.split('\n')[0])


with open("/home/alyssa/TCC/captions_train2017_pt.txt", 'r') as f:
		annotations = json.load(f)

first = True
with open("right_annots_val.txt", 'w') as outfile:
	for ids in ids_val:
		outfile.write(ids+"\n")
		for annot in annotations['annotations']:
			annotid = '%012d' % (annot['image_id'])
			if annotid == ids:
				caption = annot['caption'] + ' <end>'
				outfile.write(caption+"\n")
	

"""
#for all annotation, associate caption with respective image path (path+imageid.jpg)
for annot in annotations['annotations']:
	annotid = '%012d' % (annot['image_id'])
	if not (annotid in ids):
		continue
	#if len(all_captions) == 5:
	#	break
	if len(annot['caption'].split(" ")) > 15:
		continue

	###
	caption = + annot['caption'] + ' <end>'
	image_id = annot['image_id']
	
	full_coco_image_path = "/home/mathlima/pastaA/imageCaptioning_yolo/npy/" + '%012d.jpg' % (image_id)
	#full_coco_image_path = imageFilePath + '%012d.jpg' % (image_id)

	all_img_name_vector.append(full_coco_image_path)
	all_captions.append(caption)
"""