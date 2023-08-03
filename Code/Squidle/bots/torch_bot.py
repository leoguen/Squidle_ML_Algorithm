import torch
import torch.nn.functional as F
import os
from sqapi.helpers import create_parser
from sqapi.annotate import Annotator
from sqapi.request import query_filter as qf
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
from prediction_model import KelpClassifier

DEFAULT_CROP_PERC = 0.18

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


class TorchBOT(Annotator):

    def __init__(self, model_path: str = '/pvol/logs/Eck_cross_validation_1/perc_18/lightning_logs/version_0/checkpoints/epoch=49-step=32550.ckpt', crop_perc: float = DEFAULT_CROP_PERC, 
                **kwargs: object) -> object:
        """
        Uses pytorch to run a pytorch model
        :param model_path: the path of the pytorch model
        :param crop_perc: defines the patch size
        :param network: the network to use for the model
        """

        super().__init__(**kwargs)
        self.model = KelpClassifier.load_from_checkpoint(
            model_path,
            optimizer="AdamW",
            backbone_name='inception_v3',
            no_filters=0,
        )

        # Instantiate the Trainer with specific settings
        acc_val = 'cpu'
        if torch.cuda.is_available(): acc_val = 'gpu'
        
        self.trainer = Trainer(
            accelerator=acc_val,
            logger=False,
            num_sanity_val_steps=0,
            precision=16,
        )
        self.model.eval()
        self.crop_perc = crop_perc or DEFAULT_CROP_PERC

    def predict(self, input_data):
        # Create a SQDataset object with the cropped image for classification
        test_set = SQDataset(input_data)
        # Create a DataLoader for the test_set with batch size 1
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())
        with torch.no_grad():
            # Predict the label using the Inception-based model through the Trainer
            results = (self.trainer.predict(self.model, dataloaders=test_loader))
            #results = self.predict(test_loader)
            print(results)
            print(results[0][2])
            classifier_code = results[0][2] # "1" means object of interest
            prob = results[0][3]
        return classifier_code, prob

    def get_patch(self, x, y, mediaobj):
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

        return cropped_img

    def get_crop_points(self, x, y, original_image, img_size):
        # Calculate the crop points for the specified x-y position to extract a region of interest around the point
        x_img, y_img = original_image.size
        crop_dist = img_size / 2
        if x - crop_dist < 0: x0 = 0
        else: x0 = x - crop_dist

        if y - crop_dist < 0: y0 = 0
        else: y0 = y - crop_dist

        if x + crop_dist > x_img: x1 = x_img
        else: x1 = x + crop_dist

        if y + crop_dist > y_img: y1 = y_img
        else: y1 = y + crop_dist

        return int(x0), int(x1), int(y0), int(y1)

    def classify_point(self, mediaobj, x, y, t):
        """ returns: classifier_code, prob """
        patch_img = self.get_patch(x, y, mediaobj)
        classifier_code, prob = self.predict(patch_img)
        #predictions = predictions[0]

        # prediction in descending probability, and get the index of the max probability to equate to the class label
        #top_k = predictions.argsort()[-3:][::-1]
        #classifier_code = top_k[0]
        #prob = predictions[top_k[0]]
        return classifier_code, float(prob)

if __name__ == '__main__':

    # Running `bot = cli_init(RandoBOT)` would normally do all the steps below and initialise the class,
    # but in this instance we cant to add some extra commandline arguments to decide what annotation_sets to process

    # Get the cli arguments from the Class __init__ function signatures
    parser = create_parser(TorchBOT)

    # Add some additional custom cli args not related to the model
    parser.add_argument('--annotation_set_id', help="Process specific annotation_set", type=int, default = 8322)
    parser.add_argument('--user_group_id', help="Process all annotation_sets contained in a specific user_group", type=int)
    parser.add_argument('--affiliation_group_id', help="Process all annotation_sets contained in a specific Affiliation", type=int)
    parser.add_argument('--after_date', help="Process all annotation_sets after a date YYYY-MM-DD", type=str)
    parser.add_argument('--media_count_max', help="Filter annotation_sets that have less than a specific number of media objects", type=int)
    
    args = parser.parse_args()
    # Set the host, API key, and label map file for the bot
    args.host = 'https://staging.squidle.org'
    args.api_key = 'b113e06bfdd260844b3697be5659f9cd19beebe15231bd12fee3a979'
    args.label_map_file = '/home/ubuntu/Documents/IMAS/Code/Squidle/bots/kelp_bot_label_map.json'

    bot = TorchBOT(**vars(args))

    # Initialise annotation_set request using sqapi instance in Annotator class
    r = bot.sqapi.get("/api/annotation_set")

    # Only return annotation_sets that do not already have suggestions from this user
    r.filter_not(qf("children", "any", val=qf("user_id", "eq", bot.sqapi.current_user.get("id"))))

    # Filter annotation sets based on ID
    if args.annotation_set_id:
        r.filter("id", "eq", args.annotation_set_id)

    # Constrain date ranges to annotation_sets ceated after a specific date
    if args.after_date:
        r.filter("created_at", "gt", args.after_date)

    # Filter annotation_sets based on a user group
    if args.user_group_id:
        r.filter("usergroups", "any", val=qf("id", "eq", args.user_group_id))

    if args.affiliation_group_id:
        r.filter("user", "has", val=qf("affiliations_usergroups", "any", val=qf("group_id", "eq", args.affiliation_group_id)))

    if args.media_count_max:
        r.filter("media_count", "lte", args.media_count_max)

    # Start the bot in a loop that polls at a defined interval
    bot.start(r)
