import cv2
import os
import numpy as np

directory = './cat'
directory_ = './cat/cut'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)

        img = cv2.imread(f)
        img_ = img[400:600,150:400]
        f_ = directory_ + '/'+str(filename)
        print(f_)
        cv2.imwrite(f_,img_)




