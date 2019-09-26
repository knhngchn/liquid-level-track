from __future__ import print_function, division
import sys
import os
import torch
#import pandas as pd
import numpy as np
import random
import csv
from six import raise_from

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img_name = self.image_names[image_index]
        # while int(img_name[-8:-4]) <= 525:
        #     print(int(img_name[-8:-4]))
        #     image_index = np.random.randint(0, 2000)
        #     img_name = self.image_names[image_index]
        # print(img_name)
        img = skimage.io.imread(img_name)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        #print ()
        image = Image.open(self.image_names[image_index].replace("/media/external/","/home/kuanhung/PycharmProjects/pytorch-retinanet/726data/"))
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
    #print (max_width, max_height)
    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annots, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, min_side=900, max_side=1200, jit_translate=False):#608 1024 #1200 1600 #900 1200
        self.min_side = min_side
        self.max_side = max_side
        self.jit_translate = jit_translate

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side
        #print ("scale===================>", scale)
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image,
            (int(round(rows * scale)), int(round((cols * scale)))),
            mode='reflect',
            anti_aliasing=True,
        )
        rows, cols, cns = image.shape
        #print ("resizer =========>", image.shape)
        shift_x = 0
        shift_y = 0
        if self.jit_translate:
            pad_h = -(rows % 32)
            pad_w = -(cols % 32)
            pad_h = pad_h + 64 if pad_h != 0 else 32
            pad_w = pad_w + 64 if pad_w != 0 else 32
            shift_y = np.random.randint(pad_h)
            shift_x = np.random.randint(pad_w)
            new_image = np.zeros((rows+pad_h, cols+pad_w, cns), np.float32)
            new_image[shift_y:shift_y+rows, shift_x:shift_x+cols, :] = image.astype(np.float32)
        else:
            pad_h = 32 - rows % 32
            pad_w = 32 - cols % 32
            new_image = np.zeros((rows+pad_h, cols+pad_w, cns), np.float32)
            new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale
        annots[:, 0] += shift_x
        annots[:, 1] += shift_y
        annots[:, 2] += shift_x
        annots[:, 3] += shift_y

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

'''
def rotatepoint(p, R):
        point = np.zeros((3, 1))
        point[0] = p[0]
        point[1] = p[1]
        point[2] = 1

        new_point = R.dot(point)

        p[0] = new_point[0]

        p[1] = new_point[1]
        return p
        
def rotate_bound(image, angle, bordervalue):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bordervalue), M
'''

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, flip_x=0.5, jit_bright=0, jit_saturation=0):
        self.flip_x = flip_x
        self.jit_color = (jit_bright > 0 or jit_saturation > 0)
        self.jit_bright = jit_bright
        self.jit_saturation = jit_saturation

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        
        if np.random.rand() < self.flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
        '''
        if np.random.rand() < 0.5:
            dice = random.random()
            if dice > 0.5:
               degree = 90#(dice - 0.5) * 2 * 90
            else:
               degree = -90
            #print (annots.shape[0])
            for i in range(annots.shape[0]):
              image, R = rotate_bound(image.copy(), np.copy(degree), (0,0,0))
              a = rotatepoint([annots[i, 0].copy(), annots[i, 1].copy()], R)
              b = rotatepoint([annots[i, 2].copy(), annots[i, 3].copy()], R)
              annots[i, 0] = a[0]
              annots[i, 1] = a[1]
              annots[i, 2] = b[0]
              annots[i, 3] = b[1]
              #cv2.rectangle(image, (int(annots[i, 0]), int(annots[i, 1])), (int(annots[i, 2]), int(annots[i, 3])), (0, 255, 0), 2)
              #plt.imshow(image)
              #plt.show()
        '''
        if self.jit_color:
            image_hsv = skimage.color.rgb2hsv(image)
            b_scale = np.random.uniform(1 - self.jit_bright, 1 + self.jit_bright)
            s_scale = np.random.uniform(1 - self.jit_saturation, 1 + self.jit_saturation)
            image_hsv[..., 1] *= s_scale
            image_hsv[..., 2] *= b_scale
            image_hsv = np.clip(image_hsv, 0, 1)
            image = skimage.color.hsv2rgb(image_hsv)

        return {'img': image, 'annot': annots}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {
            'img': ((image.astype(np.float32)[:, :, :3] - self.mean) / self.std),
            'annot': annots,
        }


class UnNormalizer(object):

    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.groups = self.group_images()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
