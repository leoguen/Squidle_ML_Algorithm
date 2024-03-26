# Code relating to the paper Harnessing the power of SQUIDLE+ to develop flexible machine learning models
This code uses PyTorch to train an inception_v3 model on a dataset created using the Squidle translation tool.

This README is used to explain the workflow on how to create an individual trainingset and train a model. The libraries needed to run this experiment are saved in the requirements.txt. You can find the files in the directory "Code/PyTorch/", a GPU with cuda is strongly recommended. 

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

The easiest way to run this is by just calling the file together with you API code:

**python3 create_csv_list.py --api_token=1234567890**

This will save the annotationset into the directory "Annotationsets". You will find your API token right on the Squidle+ start page in the "My Data" section where it says: "Hi [name]: SIGN OUT, API TOKEN, ACCOUNT".

## Second Step - Modify Annotationlist
We now have a large csv file which incorporates all accessible annotations. We could train a model on this dataset and would probably achieve sufficient results. But as pointed out in the paper, this would require huge computing resources and/or time. Thus we need to modify our annotation set to acquire a more efficient and reduced dataset, our model-dataset. This is done using the **neighbour method** (for an explanation see paper); basically we define a label-of-interest, we include all labels that belong to this label-of-interest and add them to our model-dataset. Further, if an image has at least on label-of-interest annotation we add all the other labels to our model-dataset as well. 

We do that using the *modify_annotation_set.py* for you as the user the following variables can be customized to tailor the data processing for your machine learning (ML) model training task:

+ **`neighbour`**: (*Boolean*) Determines whether to use neighbour method. If set to **True**, additional annotations not related to the class of interest will be included in the dataset.

+ **`coi`**: (*String*) Class of Interest (COI) - Specifies the label class that is of primary interest for your ML model training. You can change this to match your target class.

+ **`col_name`**: (*String*) Represents the column name used for labeling within the dataset. This will mostly be 'label_translated_name' if you are using the translation service or 'label_name' if you decide not to use it.

+ **`norm_factor`**: (*Float*) A factor used for normalization of class counts. Adjust this factor to control the balance between different classes in the resulting dataset. A factor of 1 will match the amount of COI entries with an equal amount Not-Of-Interestl-Casses (NOIC). A factor 10 would lead to 10 times as many NOIC entries as COI entries. 

+ **`save_path`**: (*String*) Where you want the file to be saved.

+ **`red_list`**: (*Boolean*) Specifies whether certain entries should be marked as red-listed. If working with lineage_names it is recommended to set this true even if you do not want to supply a red_list file. If set to "True" (which is default) the csv file will also be browsed for higher hierarchy lenage_names that might include your coi but would be handled as others category. E.g. when using the lineage *Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)*, it would delete all *Physical > Substrate > Unconsolidated (soft)*, *Physical > Substrate* and *Physical*, as those might not exclusively contain our coi.

+ **`red_list_path`**: (*String*) The path to the red_list. 

Less important but also adjustable are the following variables:

+ **`sibling`**: (*Boolean*) Indicates whether to consider sibling data. If `True`, the code will handle sibling data differently during normalization. This stems from an earlier version of the code an gives the user the opportunity to include specific species at a higher percentage rate (e.g. if you want a higher percentage of "red sponges" in your model-set set you can include them here).

+ **`sibling_list_path`**: (*String*) Defines from where the sibling_list can be loaded.

+ **`sib_factor`**: (*Float*) This factor is then used in the calculations related to siblings and increases the presence of these species in the modelset.

+ **`defined_name`**: (*String*) If you want to add a specific string to the file name.

+ **`annotationset_path`**: (*String*) Path where your annotationset is saved.


The simplest command in this case would be:
**python3 modify_annotation_set.py --annotationset_path=./Annotationsets/[YOUR_Full_Annotation_List].csv --coi='Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)'**

This file now inlcudes all the annotations that we want to use for our ML training, in the next step we need to download the images. 

## Third Step - Download Images
The *download_images.py* is a simple file that downloads the images, checks for various issues (e.g., empty images), and saves the images to a specified directory (creating a subdirectory structure that will be facilitated for the training of the ML model). It only has two arguments:

+ **`save_path`**: (*String*) Defines where the images are supposed to be saved.

+ **`csv_path`**: (*String*) Defines from where the csv file can be loaded. 

After the script execution, it performs a check to determine whether the images have already been placed in the designated folder. This allows for the convenience of using a single folder to train various types of machine learning classifiers. In cases where there is an overlap in images needed by two different classifiers, the image is downloaded only once. If an image is exclusively required for one model, it will be ignored by the other, ensuring efficient use of resources.

One easy command to run in this case is:

**python3 download_images.py --csv_path='./Annotationsets/[Your_modified_annotationset].csv'**

## Fourth Step - Train the Classifier
Now that the database is prepared and downloaded we can start training the model using the *train_model.py*. Generally the model is trained using PyTorch and the PyTorch lightning wrapper. You can find an Optuna structure in there for parameter optimization, but this is not necessary for our current training, but highly recommended if you want to play around with the parameters. Overall, the script is designed to load a dataset, train a neural network model, evaluate its performance, and optimize hyperparameters using the Optuna library. It provides flexibility in configuring various aspects of the machine learning process, such as dataset loading, model architecture, hyperparameter settings, and optimization strategy.

The script begins by loading and preprocessing our dataset in the CSV_Dataset. The dataset is split into training and validation sets, with an option for cross-validation. Then we up a neural network model (default is Inception V3, but can be any other backbone you prefer to use) for the classification task.
The script defines various hyperparameters such as learning rate, batch size, and the number of epochs for training. It uses PyTorch Lightning to train the model, specifying training and validation data loaders. The model is trained with specified hyperparameters, and metrics like accuracy and F1 score are monitored during training. The training results are logged, including hyperparameters and metrics. There is an option for early stopping based on a monitored metric. It also includes a testing phase on separate test datasets, with the results logged.

If you want to implement a training pipeline using Optuna everything is in place. A function is already called objective that serves as the objective function for Optuna to maximize. The function takes trial parameters as input and trains the model with those hyperparameters. It uses Optuna's create_study function to create a study for hyperparameter optimization. The study aims to maximize a specified metric (e.g., F1 score) and uses the MedianPruner for early stopping. The script then runs the optimization process for a specified number of trials, tuning hyperparameters. After optimization, it prints out information about the best trial and its hyperparameters.

The code can be run with the following parameters:

+ **`percent_valid_examples`**: (*Float*) The percentage of valid examples taken out of the dataset for validation purposes. Default is set to **0.1**.

+ **`percent_test_examples`**: (*Float*) The percentage of examples taken out of the dataset to create a test set. Default is **0.1**.

+ **`limit_train_batches`**: (*Float*) Limits the number of batches during training for debugging purposes. Default is **0.01**.

+ **`limit_val_batches`**: (*Float*) Limits the number of validation batches for debugging purposes. Default is **0.01**.

+ **`limit_test_batches`**: (*Float*) Limits the number of test batches for debugging purposes. Default is **0.1**.

+ **`epochs`**: (*Integer*) The number of epochs for training the machine learning model. Default value is **1**.

+ **`log_path`**: (*String*) The file path where the log files will be saved. Default path is **'/pvol/logs/'**.

+ **`log_name`**: (*String*) The name of the experiment for logging purposes. Default is **'unnamed'**.

+ **`img_path`**: (*String*) The file path to the database of images used for training. Default path is **'/pvol/Final_Eck_1_to_10_Database/Original_images'**.

+ **`csv_path`**: (*String*) The file path to the CSV file describing the images in the dataset. Default is **'/pvol/Final_Eck_1_to_10_Database/Original_images/1210907_neighbour_Sand _ mud (_2mm)_list.csv'**.

+ **`n_trials`**: (*Integer*) The number of trials that Optuna will run to optimize the model parameters. Default is **1**.

+ **`backbone`**: (*String*) The name of the model architecture to be used for transfer learning. Default is **'inception_v3'**.

+ **`real_test`**: (*Boolean*) If set to **True**, a separate dataset is used for testing. If **False**, the test set is extracted from the training set. Default is **False**.

+ **`test_img_path`**: (*String*) The file path to the database of test images. Default path is **'/pvol/Ecklonia_Testbase/NSW_Broughton/'**.

+ **`img_size`**: (*Integer*) Defines the size of the images used during training. Default size is **299**.

+ **`crop_perc`**: (*Float*) The percentage of the image size that is used for cropping images. Default is **0.16**.

+ **`batch_size`**: (*Integer*) The batch size used for training the model. Generally the higher the better, but it might max out your memory so play around with it and find the best suitable value for your setup. Default is **32**.

+ **`label_name`**: (*String*) The name of the label used in the CSV file to describe the dataset. Default is **'Physical > Substrate > Unconsolidated (soft) > Sand / mud (<2mm)'**.

+ **`cross_validation`**: (*Integer*) Defines the number of sets the dataset is divided into for cross-validation purposes. Default is **0**.

+ **`col_name`**: (*String*) Name of the column that should be used for filtering, such as "label_name", "translated_label_name", "lineage_name". Default is **'label_translated_lineage_names'**.

+ **`grid_search`**: (*Boolean*) If True: a grid search from 1 to 99 percent in 5 percent increment.

Let's go through some possible commands to use this code. Before I run a long test I make sure that everything works correctly so I would for example run a command like this

**python3 train_model.py --limit_train_batches=0.1 --limit_val_batches=0.1 --limit_test_batches=0.1 --log_name=Sand_Mud_test --epochs=2 --batch_size=128 --csv_path=/pvol/Final_Eck_1_to_10_Database/Original_images/1210907_neighbour_Sand _ mud (_2mm)_list.csv**

Where limit_XXX_batches reduces the dataset to the percentage you plugin (e.g. 0.1 = 10%), the data is just randomly selected if you use this option. The code expects the csv file to be in the directory above the Images so please follow this structure if possible. The *download_images.py* creates this structure for you anyways so just move your csv file there, the structure should look like this: 

original_images/
│
├── YOUR_DATA.csv       # Your dataset CSV file
│
└── All_Images/         # Subdirectory containing all images

Once that run is finished you can check the output using the tensorboard file. See the tensorboard documentation on different ways to do this, I personally like the VS Code implementation. If that looks all good you can run the code with the correct setup. I personally run now a grid search over different patch percentages to see which performs best for my coi (See the paper for an elaborate explanation) the implementation here is 5% increments from 1 to 99% if you want to adapt this you have to change it in the code. I like to do a rough search and then naarow it down in 2% increments. 

**python3 train_model.py --grid_search --limit_train_batches=0.1 --limit_val_batches=0.1 --limit_test_batches=0.1 --log_name=Sand_Mud_test --epochs=10 --batch_size=128 --csv_path=/pvol/Final_Eck_1_to_10_Database/Original_images/1210907_neighbour_Sand _ mud (_2mm)_list.csv**

Once that is done I run a full 50 epoch training.

**python3 train_model.py --limit_train_batches=1.0 --limit_val_batches=1.0 --limit_test_batches=1.0 --log_name=Sand_Mud_Grid --epochs=50 --batch_size=128 --csv_path=/pvol/Final_Eck_1_to_10_Database/Original_images/1210907_neighbour_Sand _ mud (_2mm)_list.csv**

If you need crossvalidation for a publication that is also possible using the **`cross_validation`** flag. Be aware you cannot run  **`cross_validation`** and  **`grid_search`** at the same time.

**python3 train_model.py --crossvalidation=5 --limit_train_batches=1.0 --limit_val_batches=1.0 --limit_test_batches=1.0 --log_name=Sand_Mud_Grid --epochs=50 --batch_size=128 --csv_path=/pvol/Final_Eck_1_to_10_Database/Original_images/1210907_neighbour_Sand _ mud (_2mm)_list.csv**

Edit code to correctly do crossvalidation (correct percentage split). 

Then link to the data server code.
