Procedure to train model:
1. Download complete Annotation set (can be done using SQ directory)
2. Run modify_annotation_set.py on the Annotation set. This deletes rows that are bad formated or have missing entries and creates a new correct dataset. 
3. Run crop_and_download_images.py this generates the directory structure and downloads the images accordingly into the directories. 
4. Run the pytorch_prototye.py to train the model on the images. 
 
