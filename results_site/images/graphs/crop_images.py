import glob
import numpy as np
from PIL import Image
image_list = []
filenames=[]
for filename in glob.glob('./*.png'): 
    im=Image.open(filename)
    image_list.append(im)
    filenames.append(filename)
a=0
c=[]
for i in range(0,len(image_list)):
    image_list[i] = image_list[i].crop((190, 90, 1680, 940))
    c.append(image_list[i])
    c[i].save(filenames[i])