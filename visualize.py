import numpy as np
import torchvision
import time
import os
import time
import argparse
import glob
import torch.nn as nn
import sys
import csv
import cv2

import skimage.io
import skimage.transform
import skimage.color
import skimage
import matplotlib.cm as cm
import torch

import model


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise ValueError('line {}: format should be \'class_name,class_id\''.format(line))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.
    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
parser.add_argument('--img_prefix', help='Prefix path of images')
parser.add_argument('--txt_prefix', help='Prefix path of ground truth', default=None)
parser.add_argument('--start_frame', type=int, default=0, help='Starting frame id')
parser.add_argument('--step_frame', type=int, default=2, help='Frame step')
parser.add_argument('--model', help='Path to model (.pt) file.', default=None)
parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
parser.add_argument('--state_dict', help='Model state_dict path', default=None)
parser.add_argument('--conf_thres', type=float, default=0.85, help='conf threshold for testing')
parser.add_argument('--nms_thres', type=float, default=0.5, help='nms threshold for testing')
parser.add_argument('--dump_dir', help='Path to dump result images')
parser = parser.parse_args()

with open_for_csv(parser.csv_classes) as file:
    classes = load_classes(csv.reader(file, delimiter=','))

labels = {}
for key, value in classes.items():
    labels[value] = key

if parser.model:
    retinanet = torch.load(parser.model)
elif parser.state_dict:
    if parser.depth not in [18, 34, 50, 101, 152, 59]:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    Model = getattr(model, 'ResNet%d' % parser.depth)
    retinanet = Model(num_classes=len(classes), pretrained=True)
    # Model = getattr(model, 'ResNet%d' % parser.depth)
    # retinanet = Model(BatchNorm=nn.BatchNorm2d, pretrained=True, num_classes=3, output_stride=16) #dataset.num_classes()
    retinanet.load_state_dict(torch.load(parser.state_dict), strict=False)
    print('num of classes: %d' % len(classes))

retinanet = retinanet.cuda()
retinanet.eval()
retinanet.freeze_bn()

img_mean = np.array([[[0.485, 0.456, 0.406]]])
img_std = np.array([[[0.229, 0.224, 0.225]]])


def frame2tensor(frame):
    # Resize

    min_side = 1200  # 608
    max_side = 1600  # 1024
    rows, cols, cns = frame.shape

    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    new_shape = (int(round(rows * scale)), int(round((cols * scale))))
    frame = skimage.transform.resize(frame, new_shape, mode='reflect', anti_aliasing=True, )

    # Nomalize
    frame = ((frame[:, :, :3] - img_mean) / img_std)

    # Padding
    rows, cols, cns = frame.shape
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = frame.astype(np.float32)

    return torch.from_numpy(new_image.transpose([2, 0, 1])).unsqueeze(0), (rows, cols)
    # return torch.from_numpy(frame.transpose([2, 0, 1])).unsqueeze(0), (rows, cols)


def draw_caption(image, box, caption, caption2=None):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(image, caption, (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    if caption2:
        cv2.putText(image, caption2, (b[0], b[1] + 21), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        cv2.putText(image, caption2, (b[0], b[1] + 21), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


def draw_ground_truth(image, ori_shape, txt_path):
    if not os.path.isfile(txt_path):
        print("open txt fail")
        return
    with open(txt_path) as f:
        gt = [line.split() for line in f]

    # for cls_id, box in gt:
    #     cls_id = int(cls_id)
    #     label_name = labels[cls_id]
    #     cx = float(box[0]) * ori_shape[1]
    #     cy = float(box[1]) * ori_shape[0]
    #     w = float(box[2]) * ori_shape[1]
    #     h = float(box[3]) * ori_shape[0]
    #     x1 = int(round(cx - w / 2))
    #     y1 = int(round(cy - h / 2))
    #     x2 = int(round(cx + w / 2))
    #     y2 = int(round(cy + h / 2))
    #     cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=5)

    for entry in gt:
        cx = ((float(entry[0]) + float(entry[2]))/2) * ori_shape[1]
        cy = ((float(entry[1]) + float(entry[5]))/2) * ori_shape[0]
        w = (float(entry[2]) - float(entry[0])) * ori_shape[1]
        h = (float(entry[5]) - float(entry[1])) * ori_shape[0]
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        # x1 = int(entry[0].split('.')[0])
        # y1 = int(entry[1].split('.')[0])
        # x2 = int(entry[4].split('.')[0])
        # y2 = int(entry[5].split('.')[0])

        label_name = entry[8]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=5)


i_frame = parser.start_frame
o_frame = 0
colors = cm.rainbow(np.linspace(0, 1, 20))[:, :3]
colors = [tuple([int(v * 255) for v in c[::-1]]) for c in colors]
num_human = 0
while True:
    human_loc = []
    # input_path = '%s%d.png' % (parser.img_prefix, i_frame)
    try:
        input_path = glob.glob(os.path.join(parser.img_prefix, '*'))[i_frame]
    except IndexError:
        print('done')
        break
    # output_path = os.path.join(parser.dump_dir, '%d.png' % o_frame)
    output_path = os.path.join(parser.dump_dir, 'test_'+str(o_frame)+'.png')

    if not os.path.isfile(input_path):
        break
    frame = skimage.io.imread(input_path)

    with torch.no_grad():
        x, ori_shape = frame2tensor(frame)
        # print (x.shape)
        st = time.time()

        scores, classification, transformed_anchors = retinanet(
            x.cuda().float(),
            conf_thres=parser.conf_thres, nms_thres=parser.nms_thres,
        )
        # print(scores)
        # print(classification)
        # print(transformed_anchors)
    print('{} / Elapsed time: {}'.format(output_path, time.time() - st))

    # The shape is not the same as frame
    img = np.transpose(np.array(x[0]), (1, 2, 0)).astype(np.float32)
    img = (img * img_std + img_mean) * 255
    img = img[..., ::-1]  # To BGR format
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8).copy()

    # if parser.txt_prefix:
    #     txt_path = parser.txt_prefix
    #     draw_ground_truth(img, ori_shape, txt_path)

    # txt_path = os.path.splitext(input_path)[0] + '.txt'
    # txt_path = txt_path.replace('images', 'labelTxt')
    # draw_ground_truth(img, ori_shape, txt_path)

    for j in range(len(scores)):
        bbox = transformed_anchors[j, :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        cls_id = int(classification[j])
        label_name = labels[cls_id]
        # d, describe = distance(x1,y1,x2,y2,label_name)
        # if label_name == "human":#save human image
        #    #human_loc.append([x1,y1,x2,y2])
        #    #human_path = os.path.join(parser.dump_dir, '%d.jpg' % num_human)
        #    #new = perspective_coordinate_transform(img, ((y1+y2)/2,(x1+x2)/2))
        #    #cv2.imwrite(human_path, new[:,150:450,:]) #img[y1:y2, x1:x2,:]
        #    #num_human = num_human + 1
        #    cv2.rectangle(img, (x1, y1), (x2, y2), color=colors[cls_id], thickness=2)
        #    draw_caption(
        #     img,
        #     (x1, y1, x2, y2),
        #     label_name + "_" , '%2.2f' % (100 * float(scores[j])) #+ describe
        #    )
        cv2.rectangle(img, (x1, y1), (x2, y2), color=colors[cls_id], thickness=2)
        draw_caption(img, (x1, y1, x2, y2), label_name + "_", '%2.2f' % (100 * float(scores[j])))
        print('%-15s %.4f' % (label_name, float(scores[j])))

    # print('img_path: ', input_path)
    # print('txt_path: ', txt_path)

    cv2.imwrite(output_path, img)  # save whole image
    # np.save("./human_loc/human"+str(i_frame)+".npy", human_loc)
    i_frame += parser.step_frame
    o_frame += 1

