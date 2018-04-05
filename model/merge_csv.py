import pandas as pd

coco_train_dataframe = pd.read_csv('/scratch/at3577/coco_train/coco_train_labels.csv')

new_train_dataframe = pd.read_csv('/home/at3577/mobile-vision/models/research/object_detection/object_detection_data/train_labels.csv')

merged_train_dataframe = pd.concat([coco_train_dataframe, new_train_dataframe])

column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
merged_train_dataframe.to_csv('merged_train_labels.csv', columns=column_name, index=False)

coco_test_dataframe = pd.read_csv('/scratch/at3577/coco_train/coco_val_labels.csv')

new_test_dataframe = pd.read_csv('/home/at3577/mobile-vision/models/research/object_detection/object_detection_data/test_labels.csv')

merged_test_dataframe = pd.concat([coco_test_dataframe, new_test_dataframe])

merged_test_dataframe.to_csv('merged_test_labels.csv', columns=column_name, index=False)

