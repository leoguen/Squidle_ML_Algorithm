import os
from os import listdir
from os.path import isfile, join
from fnmatch import fnmatch
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
root = HERE
pattern = "*.jpg"

img_size = input('Image size:')
img_list = []
COUNTER = 0
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            #print(os.path.join(path, name))
            img = cv2.imread(os.path.join(path, name))
            dimensions = img.shape
            if dimensions != (img_size, img_size, 3):
                COUNTER += 1
                print('Number of messed up images: {} format {}'.
                format(COUNTER, dimensions))
                img_list.append(os.path.join(path, name))
COUNTER = 0
res = input('Do you want to remove {} files because of wrong size [y/n]?'.format(COUNTER))
if res == 'y' or 'Y':
    for i in img_list:
        os.remove(i)
        print('{} Deleted image {}'.format(COUNTER, i))
        COUNTER += 1
else:
    print('Images are not being deleted!')

