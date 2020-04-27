import time
import os
import sys

toolbar_width = 40

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))

f = open("captions_train2017_pt.txt","r") 
fTxt = f.read()
f.close()

try:
    os.remove("extractedCaptions.txt")
except:
    pass

annotation = fTxt.split("\"annotations\"")[1]
captionSec = annotation.split("},{")

print(captionSec[1])

extractedCaptions = ""
counter = 30000

t1 = time.time()

f = open("extractedCaptions.txt","a")

for caption in captionSec:
   # imageID = caption.split("\"image_id\"")[1]
   # imageID = imageID.split(",")[0]
    caption = caption.split("\"caption\"")[1]

    caption = caption.replace("\"}","")
    caption = caption.replace("]}","")
    caption = caption.replace(".","")
    caption = caption.replace(": ","")
    caption = caption.replace("\"","")

    

    #extractedCaptions = extractedCaptions + "\n" + caption
    counter = counter-1
    #f.write(caption+" "+imageID+"\n")
    f.write(caption+"\n")
    if (counter == 0):
        counter = 30000
        #extractedCaptions = ""
        sys.stdout.write("*")
        sys.stdout.flush()

f.close()
        

sys.stdout.write("]\n")

t2 = time.time()


elapsed_time = t2-t1
print(f"\nElapsed time：{elapsed_time}")