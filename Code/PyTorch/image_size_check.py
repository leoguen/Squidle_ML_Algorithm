import os
from os import listdir
from os.path import isfile, join
from fnmatch import fnmatch
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
root = HERE
pattern = "*.jpg"


COUNTER = 0
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            #print(os.path.join(path, name))
            img = cv2.imread(os.path.join(path, name))
            dimensions = img.shape
            if dimensions != (24, 24, 3):
                COUNTER += 1
                print('Number of messed up images: {} format {}'.
                format(COUNTER, dimensions))
                os.remove(os.path.join(path, name))

print('Deleted {} files because wrong size'.format(COUNTER))