# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:02:11 2019

@author: AI team
"""

import base64
from io import BytesIO

import PIL.Image
import numpy as np
import requests
from django.conf import settings
from keras.preprocessing import image


class BaseProcessing:
    file = None
    tf_serving = None

    def __init__(self, file):
        self.file = file

    @staticmethod
    def decode_image(encoded_image):
        # Decode image from base64
        return BytesIO(base64.b64decode(encoded_image))

    def get_tf_serving_pred(self, payload):
        # Get predictions from TensorFlow Serving server
        url = self.get_tf_serving_url()
        r = requests.post(url, json=payload)
        return r.json()

    def get_tf_serving_url(self):
        # TF serving url
        raise NotImplementedError

    def preprocess_image(self):
        # Preprocess image
        raise NotImplementedError

    @staticmethod
    def get_payload(img):
        # Get json for sending to TF serving server
        raise NotImplementedError

    def processing(self):
        # File processing
        raise NotImplementedError

    def get_tf_serving(self):
        if not self.tf_serving:
            raise Exception('TF serving url not found')
        return self.tf_serving


class ProcessingImageClassifier(BaseProcessing):
    tf_serving = settings.TF_SERVING_IMAGE_CLASSIFIER_URL

    def get_tf_serving_url(self):
        return f'{self.get_tf_serving()}/v1/models/image_classifier:predict'

    def preprocess_image(self):
        target_size = (28,28)
        img = image.img_to_array(image.load_img(self.file, target_size=target_size)) / 255.
        img = img[:,:,1]
        return img.reshape(1,*target_size,1)

    @staticmethod
    def get_payload(img):
        return {'instances': [{'input_image': img.tolist()}]}

    @staticmethod
    def decode_predictions(pred):
        # Decoding the response
        # decode_predictions(preds, top=3) by default gives top 3 results
        return [(str(i), i, i) for i in pred['predictions'][0]]

    def processing(self):
        img = self.preprocess_image()
        payload = self.get_payload(img)
        pred = self.get_tf_serving_pred(payload)

        return self.decode_predictions(pred)
