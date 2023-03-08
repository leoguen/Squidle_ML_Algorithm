Procedure to train model:
1. Use Create_csv_list.py to create a large csv file with all annotationssets that are accessible to you.
This is a Python script that defines a class called create_csv_list which is used to download and save data from a web API in CSV format.Here is a brief description of each method:

load_token(): This method loads an API token from a text file and returns it. The token is used to authenticate the user when requesting data from the API.

get_annotation_id(): This method uses the API token to retrieve a dataset from Squidle. The dataset contains all accessible annotation sets. The method parses the JSON response to extract the id field for each annotation set and returns a list of ids.

get_annotation_set(): This method uses the API token and the id list to retrieve the annotations for each id. For each annotation set, the method sends a GET request to the API to download a CSV file containing all the annotations. The CSV file is then saved to disk.



2. Run modify_annotation_set.py on the Annotation set. This code defines a Python class named "modify_annotation_set" that contains several methods for manipulating annotation data stored in CSV files. The methods defined in the class are as follows:

init: Constructor method that initializes two instance variables, self.sibling and self.sib_factor, to False and 0.3, respectively. This variable can be used to enhance the number of entries of a specific group.

delete_entries: This method takes the earlier created annotation set (see 1.), csv_file_df, and two strings, label and value, as input. It deletes all rows from csv_file_df where the value in the "label" column is equal to the input "value". The method then returns the modified DataFrame object. This can be used for example if the label is "Unknown" or other not helpful labels.

delete_review: This method takes a Pandas DataFrame object, csv_file_df, as input. It uses the delete_entries method to delete all rows from csv_file_df where the value in the "label_name" column is contained in a CSV file named "red_list.csv". It also deletes all rows from csv_file_df where the value in the "tag_names" column is equal to "Flagged For Review". The method then saves the modified DataFrame object to a new CSV file named "review.csv" located at the same directory as the input file, and returns the modified DataFrame object.


create_adap_set: The method uses the inputs to create a new DataFrame object that has a modified distribution of the entries in csv_file_df. It first removes all entries from csv_file_df where the value in the "label_name" column is equal to "Ecklonia radiata". It then computes a normalized distribution of the entries in csv_file_df where the value in the "label_name" column is not equal to "Ecklonia radiata", and removes all entries where the normalized frequency is less than or equal to 0.1%. The method then creates a new DataFrame object that has the same labels as the entries in norm_data_df and a number of entries for each label that is proportional to the normalized frequency computed earlier. The number of entries for the "Ecklonia radiata" label is set to be equal to the input num_eck. The method then saves the modified DataFrame object to a new CSV file named "<number_of_entries>_REVIEWED_ANNOTATION_LIST.csv" located at the same directory as the input file.

3. Run crop_and_download_images.py this code defines a class crop_download_images that can be used to crop and download images from the web, given their coordinates in a CSV file. The class has several methods:

create_directory(): creates a directory with a given name if it does not already exist
create_directory_structure(): creates a directory structure with several subdirectories to store images of different types
download_images(): downloads and crops images from the web, and saves the cropped images in the appropriate directories
get_crop_points(): given the coordinates of a point on an image, computes the coordinates of the bounding box to crop around the point
crop_image(): downloads an image from the web, crops it around a given point, and returns the cropped image and its file path
The class has several instance variables:

save_path: the path where the downloaded and cropped images will be saved
here: the directory where the CSV file is located
bounding_box: the size of the bounding box to crop around a given point
list_name: the name of the CSV file
The class has two other variables that are not defined within the class:

keep_og_size: a boolean indicating whether to keep the original size of the cropped image or to resize it to the size of the bounding box
csv_file_df: a pandas dataframe containing the data from the CSV file
The class saves several files to the save_path directory:

'_Empty_Problem.csv': a CSV file containing the indices and file paths of images that were empty or could not be downloaded
'_Crop_Problem.csv': a CSV file containing the indices and file paths of images that could not be cropped
'_Saving_Problem.csv': a CSV file containing the indices and file paths of images that could not be saved.

4. Run the optuna_lightning_model.py to train the model on the images. 
 
