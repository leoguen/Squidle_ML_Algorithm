# Import necessary libraries and modules
from sqapi.annotate import Annotator
from sqapi.request import query_filter as qf
from sqapi.helpers import cli_init, create_parser
from sqapi.media import SQMediaObject
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import torch
from pytorch_lightning import Trainer

# Set paths to the pre-trained model checkpoint and the custom CNN model directory
CKPT_PATH = '/pvol/logs/Eck_cross_validation_1/perc_18/lightning_logs/version_0/checkpoints/epoch=49-step=32550.ckpt'
#CNN_DIR_PATH = '/home/ubuntu/Documents/IMAS/Code/PyTorch'
CNN_DIR_PATH = '/home/ubuntu/Documents/IMAS/Code/Squidle/bots'
# Insert the custom CNN model directory to the system path for importing the model
sys.path.insert(0, CNN_DIR_PATH)

# Import the KelpClassifier structure from optuna_lightning_model.py
#from optuna_lightning_model import KelpClassifier
from prediction_model import KelpClassifier

# Define a custom PyTorch dataset class for the bot
class SQDataset(Dataset):
    def __init__(self, img):
        # Initialize the dataset with the input image
        self.data = [img]

    def __len__(self):
        # Return the number of samples in the dataset (always 1 in this case)
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve and preprocess the image at the specified index
        img = self.data[idx]
        train_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # Convert image to tensor [0, 255] -> [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = train_transforms(img)
        class_id = 1  # Set a default class_id (not used in this code)
        return img_tensor, class_id

# Define the InceptBOT class that extends the Annotator class
class InceptBOT(Annotator):
    def __init__(self, **annotator_args):
        # Call the constructor of the base class with the provided annotator_args
        super().__init__(**annotator_args)
        self.crop_perc = float(annotator_args['crop_perc'])/100
        # Define the possible codes that the classifier can output
        self.possible_codes = ["SUB", "ECK"]

        # Initialize the Trainer for PyTorch Lightning
        print('Initializing Trainer...')
        acc_val = 'cpu'
        if torch.cuda.is_available(): acc_val = 'gpu'
        
        # Instantiate the Trainer with specific settings
        self.trainer = Trainer(
            #num_nodes=1,
            accelerator=acc_val,
            #devices=1,
            logger=False,
            #no_filters = 0,
            num_sanity_val_steps=0,
            precision=16,
        )

        # Load the pre-trained KelpClassifier model using the provided checkpoint path
        print("Loading the model...")
        self.model = KelpClassifier.load_from_checkpoint(
            CKPT_PATH,
            optimizer="AdamW",
            backbone_name='inception_v3',
            no_filters=0,
        )

    def classify_point(self, mediaobj: SQMediaObject, x, y, t):
        """
        Overridden method: predict label for x-y point
        """
        # Retrieve image data from the media object
        image_data = mediaobj.data()  # cv2 image object containing media data
        media_path = mediaobj.url  # path to media item
        # Convert the image data to a PIL Image object
        img = Image.fromarray(image_data)

        # Calculate the crop percentage and size around the specified x-y point
        crop_size = ((img.size[0] + img.size[1]) / 2) * self.crop_perc
        x = img.size[0] * x  # Center position
        y = img.size[1] * y  # Center position
        x0, x1, y0, y1 = self.get_crop_points(x, y, img, crop_size)

        # Crop the image to the specified region of interest
        cropped_img = img.crop((x0, y0, x1, y1))

        # Create a SQDataset object with the cropped image for classification
        test_set = SQDataset(cropped_img)

        # Print information about the current classification
        print(f"CLASSIFYING: {mediaobj.url} | x: {x},  y: {y},  t: {t}")

        # Create a DataLoader for the test_set with batch size 1
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())

        # Predict the label using the Inception-based model through the Trainer
        results = (self.trainer.predict(self.model, dataloaders=test_loader))
        print(results)
        print(results[0][2])
        # Extract the classifier code from the results (SUB or ECK) and the probability
        classifier_code = self.possible_codes[results[0][2]]  # if output is "1" that means crop depicts Ecklonia
        prob = results[0][3]

        # Return the classifier code and probability
        if classifier_code == "ECK":
            return classifier_code, prob
        else:
            return classifier_code, 0

    def get_crop_points(self, x, y, original_image, img_size):
        # Calculate the crop points for the specified x-y position to extract a region of interest around the point
        x_img, y_img = original_image.size
        crop_dist = img_size / 2
        if x - crop_dist < 0:
            x0 = 0
        else:
            x0 = x - crop_dist

        if y - crop_dist < 0:
            y0 = 0
        else:
            y0 = y - crop_dist

        if x + crop_dist > x_img:
            x1 = x_img
        else:
            x1 = x + crop_dist

        if y + crop_dist > y_img:
            y1 = y_img
        else:
            y1 = y + crop_dist

        return int(x0), int(x1), int(y0), int(y1)

# Main section: Initialize and run the bot
if __name__ == '__main__':
    # Create a command-line argument parser for InceptBOT
    parser = create_parser(InceptBOT)

    # Add some additional custom command-line arguments not related to the model
    parser.add_argument('--annotation_set_id', help="Process specific annotation_set", type=int, default=8322)
    parser.add_argument('--crop_perc', help="Define crop perc.", type=int, default=18)
    parser.add_argument('--user_group_id', help="Process all annotation_sets contained in a specific user_group", type=int)
    parser.add_argument('--after_date', help="Process all annotation_sets after a date YYYY-MM-DD", type=str)

    # Initialize the global COUNTER variable
    global COUNTER
    COUNTER = 0

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set the host, API key, and label map file for the bot
    args.host = 'https://staging.squidle.org'
    # Set the host, API key, and label map file for the bot
    #open text file in read mode
    text_file = open("bots/API_KEY.txt", "r")
    #read whole file to a string
    api_key = text_file.read()
    #close file
    text_file.close()
    args.label_map_file = '/home/ubuntu/Documents/IMAS/Code/Squidle/bots/rando_bot_label_map.json'

    # Initialize the InceptBOT with the parsed arguments
    bot = InceptBOT(**vars(args))

    # Initialize an annotation_set request using the sqapi instance in Annotator class
    r = bot.sqapi.get("/api/annotation_set")

    # Filter annotation sets based on ID
    if args.annotation_set_id:
        r.filter("id", "eq", args.annotation_set_id)

    # Constrain date ranges to annotation_sets created after a specific date
    if args.after_date:
        r.filter("created_at", "gt", args.after_date)

    # Filter annotation_sets based on a user group
    if args.user_group_id:
        r.filter(name="usergroups", op="any", val=dict(name="id", op="eq", val=args.user_group_id))

    # Only return annotation_sets that do not already have suggestions from this user
    r.filter_not(qf("children", "any", val=qf("user_id", "eq", bot.sqapi.current_user.get("id"))))

    # Start the bot and perform the annotations
    bot.start(r)
