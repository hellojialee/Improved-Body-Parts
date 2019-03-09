import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO


dataType='val2017'
annFile='coco/link2coco2017/annotations/instances_{}.json'.format(dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]   #　'nms' is the shortage of 'names'
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))




