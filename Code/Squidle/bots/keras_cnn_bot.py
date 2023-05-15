from keras.models import load_model
from keras.preprocessing import image
import os
import cv2
from PIL import Image
import numpy as np
import importlib
from sqapi.helpers import create_parser

from sqapi.annotate import Annotator
from sqapi.request import query_filter as qf

DEFAULT_PATCH_WIDTH=299
DEFAULT_PATCH_HEIGHT=299
DEFAULT_NETWORK = "keras.applications.inception_v3"


class KerasBOT(Annotator):
    def __init__(self, model_path: str, patch_width: int = DEFAULT_PATCH_WIDTH, patch_height: int = DEFAULT_PATCH_HEIGHT,
                 network: str = DEFAULT_NETWORK, patch_path: str = None, **kwargs: object) -> object:
        """
        Uses keras to run a tensorflow model

        :param model_path: the path of the tensorflow model
        :param patch_width: with of the patches
        :param patch_height: height of the patches
        :param network: the network to use for the model
        :param patch_path: an optional path to cache the patches (useful if doing multiple runs on the same points)
        """

        super().__init__(**kwargs)
        self.preprocess_input = getattr(importlib.import_module(network), "preprocess_input")
        # self.preprocess_input = preprocess_input
        self.model = load_model(model_path)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.patchsize = [patch_width or DEFAULT_PATCH_WIDTH , patch_height or DEFAULT_PATCH_HEIGHT]

        self.patch_path = patch_path
        if self.patch_path is not None:
            if not os.path.isdir(self.patch_path):
                os.makedirs(self.patch_path)

    def get_patch(self, x, y, patchsize, mediaobj):
        imagename = os.path.basename(mediaobj.url)    # TODO: improve this. If image names are not unique accross instruments / deployments this will break
        padsize = int((max(patchsize) - 1) / 2)
        cropfile_name = "{}_{}_{}_{}.jpg".format(imagename, x, y, patchsize)
        cropfile_path = os.path.join(self.patch_path, cropfile_name) if self.patch_path else None

        if cropfile_path is not None and os.path.isfile(cropfile_path):
            # return cached image if it exists
            # return self.prep_patch(cropfile_path)
            return image.img_to_array(Image.open(cropfile_path))
        else:
            # convert to padded coordinates

            # check if data has been padded and if not, process it
            if not mediaobj.is_processed:
                orig_image = mediaobj.data()
                image_data = mediaobj.data(cv2.copyMakeBorder(orig_image, padsize, padsize, padsize, padsize, cv2.BORDER_REFLECT_101))
            else:
                image_data = mediaobj.data()   # has already been padded, so will return padded image

            # get patch from image with padding
            x_padded = int(round(x * mediaobj.width) + padsize)
            y_padded = int(round(y * mediaobj.height) + padsize)
            crop_image = image_data[y_padded - padsize: y_padded + padsize + 1, x_padded - padsize: x_padded + padsize + 1]

            # cache image if path is set
            if cropfile_path is not None:
                cv2.imwrite(cropfile_path, crop_image)

            return crop_image

    def classify_point(self, mediaobj, x, y, t):
        """ returns: classifier_code, prob """
        patch_img = self.get_patch(x, y, self.patchsize, mediaobj)
        patch_img = self.preprocess_input(patch_img.astype(np.float32))
        predictions = self.model.predict(np.expand_dims(patch_img, axis=0), verbose=1)
        predictions = predictions[0]
        top_k = predictions.argsort()[-3:][::-1]  # prediction in descending probability
        classifier_code = top_k[0]
        prob = predictions[top_k[0]]
        return classifier_code, float(prob)


if __name__ == '__main__':
    # Running `bot = cli_init(RandoBOT)` would normally do all the steps below and initialise the class,
    # but in this instance we cant to add some extra commandline arguments to decide what annotation_sets to process

    # Get the cli arguments from the Class __init__ function signatures
    parser = create_parser(KerasBOT)

    # Add some additional custom cli args not related to the model
    parser.add_argument('--annotation_set_id', help="Process specific annotation_set", type=int)
    parser.add_argument('--user_group_id', help="Process all annotation_sets contained in a specific user_group", type=int)
    parser.add_argument('--affiliation_group_id', help="Process all annotation_sets contained in a specific Affiliation", type=int)
    parser.add_argument('--after_date', help="Process all annotation_sets after a date YYYY-MM-DD", type=str)
    parser.add_argument('--media_count_max', help="Filter annotation_sets that have less than a specific number of media objects", type=int)

    args = parser.parse_args()
    bot = KerasBOT(**vars(args))

    # Initialise annotation_set request using sqapi instance in Annotator class
    # Only return annotation_sets that do not already have suggestions from this user
    r = bot.sqapi.get("/api/annotation_set")\
        .filter_not(qf("children", "any", val=qf("user_id", "eq", bot.sqapi.current_user.get("id"))))

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