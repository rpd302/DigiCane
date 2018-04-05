from pycocotools.coco import COCO
from os import listdir
from os.path import isfile, join
import os
import pandas as pd

annotation_file = '/scratch/at3577/coco_train/annotations/instances_val2017.json'
image_folder = '/scratch/at3577/coco_train/val2017/'

coco = COCO(annotation_file=annotation_file)

category_ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,27,28,31,33,62,63,64,65,67,70,72,73,81,82]
category_dict = {'1':  'person', 
                '2':  'bicycle', 
                '3':  'car', 
                '4':  'motorcycle', 
                '5':  'airplane', 
                '6':  'bus', 
                '7':  'train', 
                '8':  'truck', 
                '9':  'boat', 
                '10':   'traffic light',
                '11':   'fire hydrant', 
                '13':   'stop sign', 
                '14':   'parking meter', 
                '15':   'bench', 
                '16':   'bird', 
                '17':   'cat', 
                '18':   'dog', 
                '19':   'horse', 
                '20':   'sheep', 
                '27':   'backpack', 
                '28':   'umbrella', 
                '31':   'handbag', 
                '33':   'suitcase', 
                '62':   'chair', 
                '63':   'couch', 
                '64':   'potted plant', 
                '65':   'bed', 
                '67':   'dining table', 
                '70':   'toilet', 
                '72':   'tv', 
                '73':   'laptop', 
                '81':   'sink', 
                '82':   'refrigerator'}



annotation_ids = coco.getAnnIds(catIds=category_ids)

annotations = coco.loadAnns(ids=annotation_ids)

image_ids = []
for annotation in annotations:
    image_ids.append(annotation['image_id'])

images = coco.loadImgs(ids=image_ids)

annotations_dict = {'filename' : [],
                    'width' : [],
                    'height': [],
                    'class' : [],
                    'xmin' : [], 
                    'ymin' : [], 
                    'xmax' : [], 
                    'ymax' : []}
                    
for index, annotation in enumerate(annotations):
    category_id = str(annotation['category_id'])
    image_filename = images[index]['file_name']
    image_width = images[index]['width']
    image_height = images[index]['height']
    object_xmin = int(annotation['bbox'][0])
    object_ymin = int(annotation['bbox'][1])
    object_xmax = int(annotation['bbox'][0] + annotation['bbox'][2])
    object_ymax = int(annotation['bbox'][1] + annotation['bbox'][3])

    annotations_dict['filename'].append(image_filename)
    annotations_dict['width'].append(image_width)
    annotations_dict['height'].append(image_height)
    annotations_dict['class'].append(category_dict[category_id])
    annotations_dict['xmin'].append(object_xmin)
    annotations_dict['ymin'].append(object_ymin)
    annotations_dict['xmax'].append(object_xmax)
    annotations_dict['ymax'].append(object_ymax)

dataframe = pd.DataFrame.from_dict(annotations_dict)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
dataframe.to_csv('coco_val_labels.csv', columns=column_name, index=False)


