# Code relating to the paper Harnessing the power of SQUIDLE+ to develop flexible machine learning models
This code uses PyTorch to train an inception_v3 model on a dataset created using the Squidle translation tool.

The first chapter of this README is used to explain the workflow on how to create an individual trainingset and train a model. Further below you can find a more elaborate description of each file. The libraries needed to run this experiment are saved in the requirements.txt. 

## First Step 
You need to run the *create_csv_list.py* file. This is file is used to browse through all the annotationsets that you have access to. To run this script you need to add the following information when running the script (all of them have defaults, so if you are unsure you only need to add the api_token): 
 + **`api_token`**: (String) path to your api token saved as a .txt file or simply your api_token as a string (can be found on Squidle+ right next to the "SIGN OUT" option in the "My Data" field)
 + **`anno_name`**: (String) path and name of the file and where it is saved
 + **`prob_name`** : (String) path and name of the file in which all unsuccefull donwloads are written and where it is saved
 + **`export_scheme`**: (String) if you leave this as it is there will be an additional columns added to you csv file in which the translated label is saved in the Squidle schema
 + **`bool_translation`**: (Boolean) defines whether the translation column is saved or not

This should save a large csv file with several thousands entries. If you have a look at the file it self you will find a setup similar to this: 

label.id | label.name | label.uuid | point.id | point.media.deployment.campaign | point.media.path_best | point_x | point_y | ... |label.translated.id | label.translated.name
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
10925 | Sand | 8d087... | 5678 | Campaign_XY | https://aws.com | 0.2 | 0.3 |...| 646 | Sand / mud (<2mm)

As you can see from this entry, Squidle+ serves as a platform that saves the links to files uploaded on a respective server and associates them with annotations. This is achieved by assigning a label to a point in the image (identifiable using the unique path). This label corresponds to a specific position in the image defined by 'point_x' and 'point_y.' Since different annotation sets use varying labeling schemas, it can be important to translate them into a unified annotation set. This information can be found in 'label.translated.id' and remains consistent throughout the entire annotation set created in this process. 

Together with the Annotationset there is a file with a similar name only ending in "_pivot_table" in the same directory. This file shows all the label (defined by their lineage) and the count of entries you have in your dataset. It could look for example like this: 

label.translated.lineage_names |  | 
--- | --- |
Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm) > Coarse sand (with shell fragments) | 329721
Biota > Macroalgae | 246020
Biota > Macroalgae > Filamentous / filiform | 121946
Biota | 119630
Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm) | 115163
Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm) > Mud / silt (<64um) | 114601

We can see in this file that we have access to 329721 *Coarse sand (with shell fragments)* entries, 246020 *Macroalgae* and 121946 *Filamentous / filiform* entries. It is very important to be aware of the lineage of the label that you want to train. I fyou would like to train for example a classifier on *Filamentous / filiform* you should probably exclude labels like *Macroalgae* as they might include *Filamentous / filiform* entries, just that in the project it was not necessary to go down to the morphospecies level. This can be done using the red_list option in the *modify_annotation.py* set. For this example we want to train a *Sand / mud (<2mm)* classifier and don't care about lower level specifications (like *Mud / silt (<64um)*). Therefore we can just include everything that has the lineage of *Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)*.

## Second Step
We now have a large csv file which incorporates all accessible annotations. We could train a model on this dataset and would probably achieve sufficient results. But as pointed out in the paper, this would require huge computing resources and/or time. Thus we need to modify our annotation set to acquire a more efficient and reduced dataset, our model-dataset. This is done using the **neighbour method** (for an explanation see paper); basically we define a label-of-interest, we include all labels that belong to this label-of-interest and add them to our model-dataset. Further, if an image has at least on label-of-interest annotation we add all the other labels to our model-dataset as well. 

We do that using the *modify_annotation_set.py* for you as the user the following variables can be customized to tailor the data processing for your machine learning (ML) model training task:

+ **`neighbour`**: (*Boolean*) Determines whether to use neighbour method. If set to **True**, additional annotations not related to the class of interest will be included in the dataset.

+ **`coi`**: (*String*) Class of Interest (COI) - Specifies the label class that is of primary interest for your ML model training. You can change this to match your target class.

+ **`row_name`**: (*String*) Represents the column name used for labeling within the dataset. This will mostly be 'label_translated_name' if you are using the translation service or 'label_name' if you decide not to use it.

+ **`norm_factor`**: (*Float*) A factor used for normalization of class counts. Adjust this factor to control the balance between different classes in the resulting dataset. A factor of 1 will match the amount of COI entries with an equal amount Not-Of-Interestl-Casses (NOIC). A factor 10 would lead to 10 times as many NOIC entries as COI entries. 

Less important but also adjustable are the following variables:

+ **`red_list`**: (*Boolean*) Specifies whether certain entries should be marked as red-listed. If **True** the entries can be inserted using a .txt file. 

+ **`sibling`**: (*Boolean*) Indicates whether to consider sibling data. If `True`, the code will handle sibling data differently during normalization. This stems from an earlier version of the code an gives the user the opportunity to include soecific species at a higher percentage rate (e.g. if you want a higher percentage of "red sponges" in your model-set set you can include them here).

+ **`sib_factor`**: (*Float*) This factor is then used in the calculations related to siblings and increases the presence of these species in the modelset.

This file 

# File Overview
## Create_csv_list.py 
Used to create a large csv file with all annotationssets that are accessible to you.
This is a Python script that defines a class called create_csv_list which is used to download and save data from a web API in CSV format.Here is a brief description of each method:

load_token(): This method loads an API token from a text file and returns it. The token is used to authenticate the user when requesting data from the API.

get_annotation_id(): This method uses the API token to retrieve a dataset from Squidle. The dataset contains all accessible annotation sets. The method parses the JSON response to extract the id field for each annotation set and returns a list of ids.

get_annotation_set(): This method uses the API token and the id list to retrieve the annotations for each id. For each annotation set, the method sends a GET request to the API to download a CSV file containing all the annotations. The CSV file is then saved to disk.


## modify_annotation_set.py 
This code defines a Python class named "modify_annotation_set" that contains several methods for manipulating annotation data stored in CSV files. The methods defined in the class are as follows:

init: Constructor method that initializes two instance variables, self.sibling and self.sib_factor, to False and 0.3, respectively. This variable can be used to enhance the number of entries of a specific group.

delete_entries: This method takes the earlier created annotation set (see 1.), csv_file_df, and two strings, label and value, as input. It deletes all rows from csv_file_df where the value in the "label" column is equal to the input "value". The method then returns the modified DataFrame object. This can be used for example if the label is "Unknown" or other not helpful labels.

delete_review: This method takes a Pandas DataFrame object, csv_file_df, as input. It uses the delete_entries method to delete all rows from csv_file_df where the value in the "label_name" column is contained in a CSV file named "red_list.csv". It also deletes all rows from csv_file_df where the value in the "tag_names" column is equal to "Flagged For Review". The method then saves the modified DataFrame object to a new CSV file named "review.csv" located at the same directory as the input file, and returns the modified DataFrame object.


create_adap_set: The method uses the inputs to create a new DataFrame object that has a modified distribution of the entries in csv_file_df. It first removes all entries from csv_file_df where the value in the "label_name" column is equal to "Class of Interest". It then computes a normalized distribution of the entries in csv_file_df where the value in the "label_name" column is not equal to "Class of Interest", and removes all entries where the normalized frequency is less than or equal to 0.1%. The method then creates a new DataFrame object that has the same labels as the entries in norm_data_df and a number of entries for each label that is proportional to the normalized frequency computed earlier. The number of entries for the "Class of Interest" label is set to be equal to the input num_eck. The method then saves the modified DataFrame object to a new CSV file named "<number_of_entries>_REVIEWED_ANNOTATION_LIST.csv" located at the same directory as the input file.

## crop_and_download_images.py 
This code defines a class crop_download_images that can be used to crop and download images from the web, given their coordinates in a CSV file. The class has several methods:

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
 

