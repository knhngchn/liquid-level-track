# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import os

import numpy as np
from keras import backend as K
from keras.layers import Input
from PIL import Image

from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image


class YOLO(object):
    _defaults = {
        "model_path": 'trained_weights_final.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'classes.csv',
        "score": 0.35,
        "iou": 0.45,
        "model_image_size": (320,224),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self._defaults["classes_path"])
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self._defaults["anchors_path"])
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self._defaults["model_path"])
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self._defaults["model_path"])
        except:
            raise ValueError("load model weight fail")

        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect(self, image):

        if self._defaults["model_image_size"] != (None, None):
            assert self._defaults["model_image_size"][0] % 32 == 0, 'Multiples of 32 required'
            assert self._defaults["model_image_size"][1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self._defaults["model_image_size"])))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        bboxes = []
        confidences = out_scores
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            bbox = [left, top, right, bottom]
            bboxes.append(bbox)

        return bboxes, confidences

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    yolo=YOLO()
    path = 'test.jpg'
    img = Image.open(path)
    boxes, scores = yolo.detect(img)
    print(boxes, scores)

    yolo.close_session()
