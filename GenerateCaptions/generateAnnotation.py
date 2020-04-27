import os
import sys
import time

try:
    os.remove("resultAnnotation.txt")
except:
    pass

buffCaptions = "\"annotations\": ["

f = open("captions_train2017.txt","r") 
fTxt = f.read()
f.close()

f = open("pt-captions.txt","r") 
ptTxt = f.read()
ptTxt = ptTxt.splitlines()
f.close()


annotation = fTxt.split("\"annotations\": [")[1]
definitions = fTxt.split("\"annotations\": [")[0]

captionSec = annotation.split("},{")

f = open("captions_train2017_pt.txt","a")
f.write(definitions)

for i, caption in enumerate(captionSec):

    if (i != 0):
        buffCaptions = buffCaptions + "{"

    buffCaptions = buffCaptions + caption.split("\"caption\": ")[0] + "\"caption\": "
    buffCaptions = buffCaptions + "\""+ ptTxt[i]+ "\""
    buffCaptions = buffCaptions + "}"

    if (i < len(captionSec)-1):
        buffCaptions = buffCaptions + ","
        

    f.write(buffCaptions)
    buffCaptions = ""

f.write("]}")
f.close()