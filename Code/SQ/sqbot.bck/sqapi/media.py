from urllib.request import urlopen

import time

import numpy as np
import cv2


class SQMediaObject:
    """

    """
    def __init__(self, url, media_type="image", media_id=None):
        """

        :param url:
        :param media_type:
        :param media_id:
        """
        self.url = url
        self.type = media_type
        self.id = media_id
        self.width = None
        self.height = None
        self.duration = None
        self.start_time = None
        self._data = None
        self._processed_data = None
        self.is_processed = False

    def url_to_image(self, url):
        """

        :param url:
        :return:
        """
        start_time = time.time()
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print("Downloaded image: {} in {} s...".format(url, time.time() - start_time))
        return image

    def data(self, processed_data=None):
        """

        :param processed_data:
        :return:
        """
        if processed_data is not None:
            self._processed_data = processed_data
            self.is_processed = True
        elif self._data is None:
            if self.type == "image":
                self._data = self.url_to_image(self.url)
                data_shape = self._data.shape
                self.height = data_shape[0]
                self.width = data_shape[1]
            else:
                raise ValueError("Unsupported media type: {}".format(self.type))
        return self._data if not self.is_processed else self._processed_data
