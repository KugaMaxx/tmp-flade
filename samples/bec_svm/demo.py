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

#     # if sample['frames'] is None: continue
#     # image = plot_projected_events(sample['frames'], sample['events'])
#     # image = plot_detection_result(image, bboxes=bboxes)
#     # cv2.imwrite(f'./detects/detect_{i}.png', image)

print("Training...")
svm = SVC(C=1.0, kernel='rbf', gamma='auto')
svm.fit(X_train, Y_train)

# validation
data_loader = FlaDE(data_path, 'test', shuffle=True)
metirc = Metric(cats=data_loader.dataset.get_cats(key='name', query=['Flame']), 
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

    metirc.update([output], [target])

stats = metirc.summarize()
print('\n'.join(f'{info}: {value:.3f}' for info, value in stats))
