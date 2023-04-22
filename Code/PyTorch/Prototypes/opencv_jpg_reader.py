import cv2
import os

# Path to the directory containing the JPEG files
directory_path = '/pvol/Final_Eck_1_to_10_Database/Original_images/All_Images/'

# Create a list to store the filenames that result in errors/warnings
error_filenames = []

# Loop over all files in the directory
for id, filename in enumerate(os.listdir(directory_path)):
    print(id, end='\r')
    # Check if the file is a JPEG file
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        
        with open(os.path.join(directory_path, filename), 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print(f"Error for file: {filename}")
            error_filenames.append(filename)
        ''' 
        else:
            imrgb = cv2.imread(os.path.join(path, file), 1)
        
        
        try:
            # Read the image using OpenCV
            image = cv2.imread(os.path.join(directory_path, filename))
        except cv2.error as e:
            # If there is an error or warning thrown by OpenCV, add the filename to the error list
            '''

# Save the list of error filenames to a text file
with open('error_filenames.txt', 'w') as f:
    f.write('\n'.join(error_filenames))