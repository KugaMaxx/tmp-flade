import time
from dataset import FlaDE
from sklearn.svm import SVC
from cocoa_flade.utils.benchmark import Metric
from cocoa_flade.utils.draw import plot_projected_events, plot_detection_result

from generator import BinaryEventContour
from extractor import FeatExtractor

import torch
from torchvision.ops import box_iou
from torchvision.ops.boxes import _box_xywh_to_xyxy


data_path = '/home/szd/workspace/tmp-flade/data/FlaDE'

# training
print("Training...")
X_train, Y_train = [], []
data_loader = FlaDE(data_path, 'train', shuffle=True, num_samples=600)
for i, (sample, target) in enumerate(data_loader):
    # create output
    output = dict()
    
    # rois generation
    gen = BinaryEventContour(target['resolution'])
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
            Y_train.append(target['labels'][ind] + 1)

svm = SVC(C=1.0, kernel='rbf', gamma='auto')
svm.fit(X_train, Y_train)

# validation
print("Validation...")
data_loader = FlaDE(data_path, 'test', shuffle=True)
metirc = Metric(cats=data_loader.dataset.get_cats(), 
                tags=data_loader.dataset.get_tags())
for i, (sample, target) in enumerate(data_loader):
    # ceate output
    output = dict()

    # roi generation
    gen = BinaryEventContour(target['resolution'])
    output['bboxes'] = gen.detect(sample)

    # feat extraction
    ext = FeatExtractor(target['resolution'])
    output['feats'] = ext.process(sample, output['bboxes'])
    
    # whether roi is empty
    if len(output['bboxes']) == 0:
        output['labels'] = []
        output['scores'] = []
    else:
        output['labels'] = [c - 1 for c in svm.predict(output['feats'])]
        output['scores'] = [1. for _ in range(len(output['labels']))]

    if 0 in output['labels']: 
        if 0 not in target['labels']:
            pass
            # import cv2
            # if sample['frames'] is None: continue
            # image = plot_projected_events(sample['frames'], sample['events'])
            # image = plot_detection_result(image, bboxes=output['bboxes'])
            # cv2.putText(image, f"{target['name']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # cv2.imwrite(f'./detects/detect_{i}.png', image)

    metirc.update([output], [target])

stats, brief = metirc.summarize()
print(brief)
