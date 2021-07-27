import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def loss(classifications, regression, anchors, annotations):
    # classification should be logit
    alpha = 0.25
    gamma = 2.0
    batch_size = classifications.shape[0]

    classification_losses = []
    regression_losses = []

    anchor = anchors[0, :, :]

    for j in range(batch_size):

        classification = classifications[j, :, :]
        annotation = annotations[j].float().cuda()

        if annotation.shape[0] == 0:
            print('no annots')
            regression_losses.append(torch.tensor(0).float().cuda())
            classification_losses.append(torch.tensor(0).float().cuda())
            continue
        
        # classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
        clf_p = F.sigmoid(classification)
        clf_p = torch.clamp(clf_p, 1e-4, 1.0 - 1e-4)
        #print ("Calculate IOU")
        IoU = calc_iou(anchors[0, :, :], annotation[:, :4]) # num_anchors x num_annotations
        #print ("Finish Calculate IOU")
        IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

        # compute the loss for classification
        targets = torch.zeros(classification.shape)
        dontcare = torch.ones(classification.shape).long()
        targets = targets.cuda()
        dontcare = dontcare.cuda()

        dontcare[torch.lt(IoU_max, 0.4), :] = 0
        #print ("CHECKPOINT")
        positive_indices = torch.ge(IoU_max, 0.5)

        num_positive_anchors = positive_indices.sum()

        assigned_annotations = annotation[IoU_argmax, :]
        
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
        dontcare[positive_indices, :] = 0

        alpha_factor = torch.ones(targets.shape).cuda() * alpha

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - clf_p, clf_p)
        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

        bce = F.binary_cross_entropy_with_logits(classification, targets, reduce=False)

        # cls_loss = focal_weight * torch.pow(bce, gamma)
        cls_loss = focal_weight * bce

        cls_loss = torch.where(torch.eq(dontcare, 0), cls_loss, torch.zeros(1).cuda())

        classification_losses.append(
            cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
        )

        # compute the loss for regression

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
        gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
        gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
        gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

        # clip widths to 1
        gt_widths  = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)

        targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
        targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
        targets_dw = torch.log(gt_widths / anchor_widths)
        targets_dh = torch.log(gt_heights / anchor_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
        targets = targets.t()

        targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
        # targets = targets/torch.Tensor([[1., 1., 1., 1.]]).cuda()


        negative_indices = 1 - positive_indices

        regression_diff = torch.abs(targets - regression[j, :, :])

        regression_diff[negative_indices, :] = 0

        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        )
        if positive_indices.sum() > 0:
            regression_losses.append(regression_loss[positive_indices, :].mean())
        else:
            regression_losses.append(torch.tensor(0).float().cuda())
        
    return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()
