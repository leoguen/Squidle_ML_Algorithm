import random
import sys
sys.path.insert(0, '/home/leo/Documents/IMAS/Code/SQ/sqbot/sqapi')
from annotate import Annotator, register_annotator_plugin


class KelpBOT(Annotator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.possible_codes = ["ECK", "ASC", "SUB"]

    def classify_point(self, mediaobj, x, y, t):
        """
        Overridden method: predict label for x-y point
        """
        # image_data = mediaobj.data()            # cv2 image object containing media data
        # media_path = mediaobj.url               # path to media item
        print(f"CLASSIFYING: {mediaobj.url} | x: {x},  y: {y},  t: {t}")
        classifier_code = random.sample(self.possible_codes, 1)[0]
        prob = round(random.random(), 2)
        return classifier_code, prob


register_annotator_plugin(KelpBOT, name="KelpBOT")