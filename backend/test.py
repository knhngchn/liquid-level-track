import time
import argparse
import sys
import numpy as np
import pdb
import os

import torch
from torchvision import datasets, models, transforms
import torchvision
import torch.nn as nn
#import pandas as pd
import matplotlib
matplotlib.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sns

import model

import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from utils import bbox_iou, compute_ap
from torch.utils.data import Dataset, DataLoader


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


parser = argparse.ArgumentParser(description='Eval mAP/recall/precision on given dataset.')

parser.add_argument('--dataset', help='Dataset type, must be csv currently.')
parser.add_argument('--csv', help='Path to file containing testing annotations (see readme)')
parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
parser.add_argument('--model', help='Model path', default=None)
parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
parser.add_argument('--state_dict', help='Model state_dict path', default=None)
parser.add_argument('--conf_thres', type=float, default=0.5, help='conf threshold for testing')
parser.add_argument('--nms_thres', type=float, default=0.5, help='nms threshold for testing')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--confusion_matrix', default=None, help='Path to output confusion matrix')

parser = parser.parse_args()


# Create the data loaders
if parser.dataset == 'csv':
    if parser.csv is None:
        raise ValueError('Must provide --csv when training on COCO,')
    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes when training on COCO,')
    dataset = CSVDataset(train_file=parser.csv, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

else:
    raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False, shuffle=False)
dataloader = DataLoader(dataset, num_workers=4, collate_fn=collater, batch_sampler=sampler)

# Create the model
if parser.model:
    retinanet = torch.load(parser.model)
elif parser.state_dict:
    # retinanet = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, num_classes=dataset.num_classes(), output_stride=16)
    # if parser.depth not in [18, 34, 50, 101, 152]:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    # Model = getattr(model, 'ResNet%d' % parser.depth)
    # retinanet = Model(num_classes=dataset.num_classes(), pretrained=True)
    Model = getattr(model, 'ResNet%d' % parser.depth)
    # retinanet = Model(BatchNorm=nn.BatchNorm2d, pretrained=True, num_classes=dataset.num_classes(), output_stride=16)
    retinanet = Model(pretrained=True, num_classes=dataset.num_classes())
    retinanet.load_state_dict(torch.load(parser.state_dict), strict=False)
retinanet = retinanet.cuda()
retinanet.training = False
retinanet.eval()
retinanet.freeze_bn()

# Testing
targets = None
APs = []
total_TP = 0
total_FP = 0
total_annot = 0
total_pred = 0
wrong_clf = 0
y_true = []
y_pred = []

# write to compute mAP
mAP_gt_path = './mAP/ground-truth/'
mAP_p_path = './mAP/predicted/'
# os.makedirs(os.path.join(mAP_gt_path,parser.state_dict.split('/')[-1]), exist_ok=True)
# os.makedirs(os.path.join(mAP_p_path,parser.state_dict.split('/')[-1]), exist_ok=True)

with open(parser.csv_classes) as f:
    id2label = [line.split(',')[0] for line in f]
print (id2label)
# conf_thres = [0.5081147849559784, 0.36635568737983704, 0.449447825551033, 0.47068123519420624, 0.45850513875484467, 0.47677767276763916, 0.3495875746011734, 0.4107544273138046, 0.46883130073547363, 0.44973841309547424, 0.4029313623905182, 0.4842960387468338, 0.46613171696662903, 0.4920552223920822, 0.4495714455842972, 0.40201660990715027, 0.5011806488037109, 0.5912651419639587, 0.45688609778881073, 0.5079730153083801]
conf_thres = [0.5,0.5,0.5]
assert(len(conf_thres) == len(id2label))
assert(id2label == ['human', 'book', 'cup'])
# assert(id2label == ['bed', 'bookcase', 'bottle', 'bow', 'cabin', 'cellphone', 'chair', 'chopstick', 'couch', 'cup', 'fridge', 'key', 'remote', 'socket_switch', 'spoon_fork', 'table', 'tissue', 'toilet', 'umbrella', 'wallet'])
heatmap = np.zeros((len(id2label) + 1, len(id2label) + 1), dtype=np.int64)


def write_mAP(data, classification, scores, transformed_anchors):
    # pdb.set_trace()

    # write to compute mAP
    gt_output_path = os.path.join(mAP_gt_path, str(ith) + '.txt')
    p_output_path = os.path.join(mAP_p_path, str(ith) + '.txt')
    gt_output = data['annot'][0].numpy()
    gt_output = np.around(np.hstack([gt_output[:, 4:], gt_output[:, :4]])).astype(int).astype(str)

    for gt_line in gt_output:
        gt_line[0] = id2label[int(gt_line[0])]

    with open(gt_output_path, 'w+') as f:
        for line in gt_output:
            f.write(' '.join(str(item) for item in line))
            f.write('\n')

    # predicted_output
    with open(p_output_path, 'w+') as f:
        for item in range(classification.size()[0]):
            write_line = (id2label[int(classification[item])], str(float(scores[item])))
            for ii in transformed_anchors[item].numpy():
                write_line = write_line + (str(int(round(ii))),)
            f.write(' '.join(write_line))
            f.write('\n')


for ith, data in enumerate(dataloader):
    # Currently batchsize == 1
    with torch.no_grad():
        imgs = data['img'].cuda().float()
        # x1 y1 x2 y2 labelid
        annotations = data['annot'][0]
        # sorted
        scores, classification, transformed_anchors = retinanet(
            imgs, conf_thres=parser.conf_thres, nms_thres=parser.nms_thres,
        )
    annotations = annotations.cpu().float()
    scores = scores.cpu().float()
    classification = classification.cpu().float()
    transformed_anchors = transformed_anchors.cpu()

    not_found = set()
    for idx in range(annotations.size(0)):
        not_found.add(idx)

    # Modified from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/test.py
    # Compute average precision for current sample
    if scores.size(0) == 0:
        # If there are no detections but there are annotations mask as zero AP
        if annotations.size(0) != 0:
            APs.append(0)
            total_annot += annotations.size(0)
            for idx in not_found:
                heatmap[int(annotations[idx, 4].item()), -1] += 1
            #write_mAP(data, classification, scores, transformed_anchors)
        continue

    correct = []

    # If no annotations add number of detections as incorrect
    if annotations.size(0) == 0:
        correct.extend([0 for _ in range(scores.size(0))])
        for conf, obj_pred in zip(scores, classification):
            obj_pred = int(obj_pred.itme())
            if conf < conf_thres[obj_pred]:
                continue
            correct.append(0)
            heatmap[-1, obj_pred] += 1
    else:
        target_boxes = annotations[:, :4]
        detected = []
        for conf, pred_bbox, obj_pred in zip(scores, transformed_anchors, classification):
            obj_pred = int(obj_pred.item())
            if conf < conf_thres[obj_pred]:
                continue
            pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)

            iou = bbox_iou(pred_bbox, target_boxes)
            best_i = int(np.argmax(iou))
            obj_annt = int(annotations[best_i, 4].item())

            if iou[best_i] > parser.iou_thres:
                if obj_pred == obj_annt and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                    not_found.remove(best_i)
                else:
                    correct.append(0)
                y_true.append(obj_annt)
                y_pred.append(obj_pred)
                heatmap[y_true[-1], y_pred[-1]] += 1
            else:
                correct.append(0)
                heatmap[-1, y_pred[-1]] += 1
        for idx in not_found:
            heatmap[int(annotations[idx, 4].item()), -1] += 1

    # write_mAP(data, classification, scores, transformed_anchors)

    # Extract true and false positives
    true_positives  = np.array(correct)
    false_positives = 1 - true_positives

    # Compute cumulative false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)

    # Compute recall and precision at all ranks
    recall    = true_positives / annotations.size(0) if annotations.size(0) else true_positives
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    now_TP        = np.sum(correct) if len(correct) else 0
    now_FP        = len(correct) - now_TP
    now_recall    = now_TP / annotations.size(0) if annotations.size(0) else 1
    now_precision = now_TP / max(len(correct), 1e-6)
    total_TP    += now_TP
    total_FP    += now_FP
    total_annot += annotations.size(0)
    total_pred  += len(correct)

    # Compute average precision
    AP = compute_ap(recall, precision)
    APs.append(AP)

    print('[%d/%d] AP: %.4f (%.4f) recall: %.4f (%.4f) precision: %.4f (%.4f) FP per image: %d (%.4f)' % (
        len(APs), len(dataset),
        AP, np.mean(APs),
        now_recall, total_TP / max(total_annot, 1e-6),
        now_precision, total_TP / max(total_pred, 1e-6),
        now_FP, total_FP / (ith + 1)
    ), flush=True)


if parser.confusion_matrix and len(y_true):
    heatmap = heatmap.astype(np.float64)
    rownorm_heatmap = heatmap / (heatmap.sum(1).reshape(-1, 1) + 1e-6)
    colnorm_heatmap = heatmap / (heatmap.sum(0) + 1e-6)
    id2label += ['no_obj']
    fig = plt.figure(figsize=(16, 12))
    sns.heatmap(colnorm_heatmap, xticklabels=id2label, yticklabels=id2label, annot=True)
    fig.savefig(parser.confusion_matrix + '_colnorm_comb_all2.jpg')

    fig = plt.figure(figsize=(16, 12))
    sns.heatmap(rownorm_heatmap, xticklabels=id2label, yticklabels=id2label, annot=True)
    fig.savefig(parser.confusion_matrix + '_rownorm_comb_all2.jpg')

