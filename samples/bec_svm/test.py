import time
from dataset import FlaDE
from sklearn.svm import SVC

from generator import BinaryEventContour
from extractor import FeatExtractor

print("Now training svm...")
st = time.time()
num_samples = 600
X_train, Y_train = [], []
for i, (sample, target) in enumerate(FlaDE('/data/Ding/FlaDE', 'train', shuffle=True, num_samples=num_samples)):
    # ROI generation
    gen = BinaryEventContour(target['resolution'])
    bboxes = gen.detect(sample)

    # Feat extraction
    ext = FeatExtractor(target['resolution'])
    feats = ext.process(sample, bboxes)

    # Construct
    # X_train.extend(feats)
    # Y_train.extend(feats)

print(time.time() - st)

# svm = SVC(C=1.0, kernel='rbf', gamma='auto')
# svm.fit(X_train, Y_train)
