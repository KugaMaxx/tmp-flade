import time
from dataset import FlaDE
from sklearn.svm import SVC
from collections import defaultdict
from cocoa_flade.utils.benchmark import Timer, Metric
from cocoa_flade.utils.draw import plot_projected_events, plot_detection_result, plot_rescaled_image

from generator import BinaryEventContour
from extractor import FeatExtractor

import cv2
import torch
from torchvision.ops import box_iou
from torchvision.ops.boxes import _box_xywh_to_xyxy
import numpy as np


data_path = '/home/szd/workspace/tmp-flade/data/FlaDE'

# training
print("Training...")
X_train, Y_train = [], []
data_loader = FlaDE(data_path, 'train', shuffle=True, num_samples=600)
for i, (sample, target) in enumerate(data_loader):
    # create output
    output = dict()
    
    # rois generation, restrict single roi because of unlabeled flying flame
    gen = BinaryEventContour(target['resolution'], candidate_num=1)
    output['bboxes'] = gen.detect(sample)

    # feat extraction
    ext = FeatExtractor(target['resolution'])
    output['feats'] = ext.process(sample, output['bboxes'])

    # whether roi is empty
    if len(output['bboxes']) == 0: continue

    # calculate ious
    output['ious'] = box_iou(_box_xywh_to_xyxy(torch.tensor(output['bboxes'])), 
                             _box_xywh_to_xyxy(torch.tensor(target['bboxes'])))
    
    # construct training input
    for feat, iou in zip(output['feats'], output['ious']):
        # search most candidate label
        ind = iou.argmax()
        if iou[ind] == 0:
            X_train.append(feat)
            Y_train.append(0)
        else:
            X_train.append(feat)
            Y_train.append(target['labels'][ind])

svm = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True)
svm.fit(X_train, Y_train)

# validation
print("Validation...")
data_loader = FlaDE(data_path, 'val', shuffle=True)
timer  = Timer()
metric = Metric(cats=data_loader.dataset.get_cats(), 
                tags=data_loader.dataset.get_tags())
for i, (sample, target) in enumerate(data_loader):
    # ceate output
    output = dict()

    with timer:
        # count number
        timer.count(1)

        # roi generation, restrict single roi because of unlabeled flying flame
        gen = BinaryEventContour(target['resolution'], candidate_num=1)
        output['bboxes'] = gen.detect(sample)

        # feat extraction
        ext = FeatExtractor(target['resolution'])
        output['feats'] = ext.process(sample, output['bboxes'])
        
        # whether roi is empty
        if len(output['bboxes']) == 0:
            output['labels'] = []
            output['scores'] = []
        else:
            output['labels'] = [c for c in svm.predict(output['feats'])]
            output['scores'] = [p.max() for p in svm.predict_proba(output['feats'])]

    # if 0 in output['labels'] and 0 in target['labels']:
    #     ind = np.array(output['labels']) == 0
    #     iou = box_iou(_box_xywh_to_xyxy(torch.tensor(output['bboxes'])),
    #                   _box_xywh_to_xyxy(torch.tensor(target['bboxes'])))
        
    #     if iou[ind, 0].min() < 0.5:
        # image = plot_projected_events(sample['frames'], sample['events'])
        # image = plot_rescaled_image(image, factor=5)
        # image = plot_detection_result(image, bboxes=output['bboxes'], labels=output['labels'], scores=output['scores'])
        # image = plot_detection_result(image, bboxes=target['bboxes'], labels=target['labels'], 
        #                               categories=data_loader.dataset.get_cats())
        # cv2.putText(image, f"{target['name']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # cv2.imwrite(f'./detects/detect_{i}.png', image)
        # print(f'./detects/detect_{i}.png', 
        #         target['name'], 
        #         _box_xywh_to_xyxy(torch.tensor([
        #             output['bboxes'][0][0] * 346,
        #             output['bboxes'][0][1] * 260,
        #             output['bboxes'][0][2] * 346,
        #             output['bboxes'][0][3] * 260
        #         ])))

    metric.update([output], [target])

t_stats, t_brief = timer.report()
print(t_brief)

m_stats, m_brief = metric.summarize()
print(m_brief)
