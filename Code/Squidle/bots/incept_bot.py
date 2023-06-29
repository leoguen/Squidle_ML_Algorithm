import random
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

CKPT_PATH = '/pvol/logs/Eck_cross_validation_1/perc_18/lightning_logs/version_0/checkpoints/epoch=49-step=32550.ckpt'
CNN_DIR_PATH = '/home/ubuntu/Documents/IMAS/Code/PyTorch'

sys.path.insert(0,CNN_DIR_PATH)

from optuna_lightning_model import KelpClassifier #Import the KelpClassifier structure from optuna_lightning_model.py

class SQDataset(Dataset):

    def __init__(self, img):
        self.data = [img]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = train_transforms(img)
        class_id = 1
        return img_tensor, class_id


class InceptBOT(Annotator):
    def __init__(self, **annotator_args):
        """
        Short description
        An example of an automated labelling bot that selects random class labels to assign to points.
        It provides terrible suggestions, however it provides a simple boiler-plate example of how to integrate a
        Machine Learning algorithm for label suggestions.
        """
        super().__init__(**annotator_args)
        self.possible_codes = ["SUB","ECK"]

        print('Initializing Trainer...')

        acc_val = 'cpu'

        if torch.cuda.is_available(): acc_val = 'gpu'
        # Instantiate the Trainer
        self.trainer = Trainer(
        #num_nodes=1,
        accelerator=acc_val,
        #devices=1,
        logger = False,
        #no_filters = 0,
        num_sanity_val_steps=0,
        precision=16,
        )

        print("Loading the model...")
        self.model = KelpClassifier.load_from_checkpoint(
        CKPT_PATH,
        LEARNING_RATE = 5e-10,
        trainer = self.trainer,
        trial = 0,
        img_size = 299,
        batch_size = 200,
        pretrained=True,
        optimizer="AdamW",
        criterion="cross_entropy",
        backbone_name = 'inception_v3',
        no_filters = 0,
        )


    def classify_point(self, mediaobj: SQMediaObject, x, y, t):
        """
        Overridden method: predict label for x-y point
        """
        image_data = mediaobj.data()            # cv2 image object containing media data
        media_path = mediaobj.url               # path to media item
        img = Image.fromarray(image_data)
        crop_perc = 0.18
        crop_size = ((img.size[0]+img.size[1])/2)*crop_perc
        x = img.size[0]*x #Center position
        y = img.size[1]*y #Center position
        x0, x1, y0, y1 = self.get_crop_points(x, y, img, crop_size)
        cropped_img = img.crop((x0, y0, x1, y1))
        #global COUNTER
        #url_name = media_path.rsplit('/', 1)[1]
        #cropped_img.save('/home/ubuntu/Documents/IMAS/Code/Squidle/bots/crop_examples/'+str(COUNTER)+'_'+url_name[1:])
        #COUNTER +=1

        #if this does not work check cv2 composition
        test_set = SQDataset(cropped_img)
        print(f"CLASSIFYING: {mediaobj.url} | x: {x},  y: {y},  t: {t}")
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())
        results = (self.trainer.predict(self.model, dataloaders=test_loader))
        print(results)
        print(results[0][2])
        #results = batch_idx, int(y[0]), int(top_class.data[0][0]), float(top_p.data[0][0])
        classifier_code = self.possible_codes[results[0][2]]
        prob = results[0][3]
        if classifier_code == "ECK":
            return classifier_code, prob
        else:
            return classifier_code, 0

    def get_crop_points(self, x, y, original_image, img_size):
        x_img, y_img = original_image.size
        crop_dist = img_size/2 
        if x - crop_dist < 0: x0 = 0
        else: x0 = x - crop_dist

        if y - crop_dist < 0: y0 = 0
        else: y0 = y - crop_dist

        if x + crop_dist > x_img: x1 = x_img
        else: x1 = x + crop_dist

        if y + crop_dist > y_img: y1 = y_img
        else: y1 = y + crop_dist

        return  int(x0), int(x1), int(y0), int(y1)
        

if __name__ == '__main__':
    # Running `bot = cli_init(RandoBOT)` would normally do all the steps below and initialise the class,
    # but in this instance we cant to add some extra commandline arguments to decide what annotation_sets to process

    # Get the cli arguments from the Class __init__ function signatures
    parser = create_parser(InceptBOT)

    # Add some additional custom cli args not related to the model
    parser.add_argument('--annotation_set_id', help="Process specific annotation_set", type=int, default=8310)
    parser.add_argument('--user_group_id', help="Process all annotation_sets contained in a specific user_group", type=int)
    parser.add_argument('--after_date', help="Process all annotation_sets after a date YYYY-MM-DD", type=str)
    global COUNTER
    COUNTER = 0
    args = parser.parse_args()
    args.host = 'https://staging.squidle.org'
    args.api_key = 'b113e06bfdd260844b3697be5659f9cd19beebe15231bd12fee3a979'
    args.label_map_file = '/home/ubuntu/Documents/IMAS/Code/Squidle/bots/rando_bot_label_map.json'
    bot = InceptBOT(**vars(args))

    # Initialise annotation_set request using sqapi instance in Annotator class
    r = bot.sqapi.get("/api/annotation_set")

    # Filter annotation sets based on ID
    if args.annotation_set_id:
        r.filter("id", "eq", args.annotation_set_id)

    # Constrain date ranges to annotation_sets ceated after a specific date
    if args.after_date:
        r.filter("created_at", "gt", args.after_date)

    # Filter annotation_sets based on a user group
    if args.user_group_id:
        r.filter(name="usergroups", op="any", val=dict(name="id", op="eq", val=args.user_group_id))

    # Only return annotation_sets that do not already have suggestions from this user
    r.filter_not(qf("children", "any", val=qf("user_id", "eq", bot.sqapi.current_user.get("id"))))

    bot.start(r)