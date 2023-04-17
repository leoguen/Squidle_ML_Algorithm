#from io import BytesIO
#from urllib.request import urlopen
from keras.models import load_model
from keras.preprocessing import image
# from keras.applications.inception_v3 import preprocess_input #, InceptionV3
import os
import cv2
from PIL import Image
import numpy as np
import importlib
# import sys

from sqapi.annotate import Annotator, SQAPIargparser, register_annotator_plugin

DEFAULT_PATCH_WIDTH=299
DEFAULT_PATCH_HEIGHT=299
DEFAULT_NETWORK = "keras.applications.inception_v3"



class KerasBOT(Annotator):
    def __init__(self, model_path, patch_width=DEFAULT_PATCH_WIDTH, patch_height=DEFAULT_PATCH_HEIGHT,
                 network=DEFAULT_NETWORK, patch_path=None, **kwargs):

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

    @classmethod
    def add_arguments(cls):
        super().add_arguments()
        SQAPIargparser.add_argument("--model_path", type=str, help="Path to Keras classifier model", required=True)
        SQAPIargparser.add_argument("--network", type=str, help=f"Network architecture package (default '{DEFAULT_NETWORK}')", default=DEFAULT_NETWORK)
        SQAPIargparser.add_argument("--patch_height", type=int, default=DEFAULT_PATCH_HEIGHT,
                                    help=f"Height of image patch (default {DEFAULT_PATCH_HEIGHT})")
        SQAPIargparser.add_argument("--patch_width", type=int,
                                    help=f"Width of image patch (default {DEFAULT_PATCH_WIDTH})",default=DEFAULT_PATCH_WIDTH)
        SQAPIargparser.add_argument("--patch_path", type=str, help="Path to cache image patches", required=False)

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


register_annotator_plugin(KerasBOT, name="KerasBOT")