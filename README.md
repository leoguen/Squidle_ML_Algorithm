# Code relating to the paper Harnessing the power of SQUIDLE+ to develop flexible machine learning models
This code uses PyTorch to train an inception_v3 model on a dataset created using the Squidle translation tool.

The first chapter of this README is used to explain the workflow on how to create an individual trainingset and train a model. Further below you can find a more elaborate description of each file. The libraries needed to run this experiment are saved in the requirements.txt. 

## First Step - Acquire full Annotationlist
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

## Second Step - Modify Annotationlist
We now have a large csv file which incorporates all accessible annotations. We could train a model on this dataset and would probably achieve sufficient results. But as pointed out in the paper, this would require huge computing resources and/or time. Thus we need to modify our annotation set to acquire a more efficient and reduced dataset, our model-dataset. This is done using the **neighbour method** (for an explanation see paper); basically we define a label-of-interest, we include all labels that belong to this label-of-interest and add them to our model-dataset. Further, if an image has at least on label-of-interest annotation we add all the other labels to our model-dataset as well. 

We do that using the *modify_annotation_set.py* for you as the user the following variables can be customized to tailor the data processing for your machine learning (ML) model training task:

+ **`neighbour`**: (*Boolean*) Determines whether to use neighbour method. If set to **True**, additional annotations not related to the class of interest will be included in the dataset.

+ **`coi`**: (*String*) Class of Interest (COI) - Specifies the label class that is of primary interest for your ML model training. You can change this to match your target class.

+ **`col_name`**: (*String*) Represents the column name used for labeling within the dataset. This will mostly be 'label_translated_name' if you are using the translation service or 'label_name' if you decide not to use it.

+ **`norm_factor`**: (*Float*) A factor used for normalization of class counts. Adjust this factor to control the balance between different classes in the resulting dataset. A factor of 1 will match the amount of COI entries with an equal amount Not-Of-Interestl-Casses (NOIC). A factor 10 would lead to 10 times as many NOIC entries as COI entries. 

+ **`save_path`**: (*String*) Where you want the file to be saved.



Less important but also adjustable are the following variables:

+ **`red_list`**: (*Boolean*) Specifies whether certain entries should be marked as red-listed. If **True** the entries can be inserted using a .txt file. 

+ **`sibling`**: (*Boolean*) Indicates whether to consider sibling data. If `True`, the code will handle sibling data differently during normalization. This stems from an earlier version of the code an gives the user the opportunity to include soecific species at a higher percentage rate (e.g. if you want a higher percentage of "red sponges" in your model-set set you can include them here).

+ **`sibling_list_path`**: (*String*) Defines from where the sibling_list can be loaded.

+ **`sib_factor`**: (*Float*) This factor is then used in the calculations related to siblings and increases the presence of these species in the modelset.

+ **`defined_name`**: (*String*) If you want to add a specific string to the file name.

This file noe inlcudes all the annotations that we want to use for our ML training, in the next step we need to download the images. 

## Third Step - Download Images
The *download_images.py* is a simple file that downloads the images, checks for various issues (e.g., empty images), and saves the images to a specified directory (creating a subdirectory structure that will be facilitated for the training of the ML model). It only has two arguments:

+ **`save_path`**: (*String*) Defines where the images are supposed to be saved.

+ **`csv_path`**: (*String*) Defines from where the csv file can be loaded. 

After the script execution, it performs a check to determine whether the images have already been placed in the designated folder. This allows for the convenience of using a single folder to train various types of machine learning classifiers. In cases where there is an overlap in images needed by two different classifiers, the image is downloaded only once. If an image is exclusively required for one model, it will be ignored by the other, ensuring efficient use of resources.
