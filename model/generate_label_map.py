labels_dict = {'person': 1,
'bicycle': 2,
'car': 3,
'motorcycle': 4,
'airplane': 5,
'bus': 6,
'train': 7,
'truck': 8,
'boat': 9,
'traffic light': 10,
'fire hydrant': 11,
'stop sign': 12,
'parking meter': 13,
'bench': 14,
'bird': 15,
'cat': 16,
'dog': 17,
'horse': 18,
'sheep': 19,
'backpack': 20,
'umbrella': 21,
'handbag': 22,
'suitcase': 23,
'chair': 24,
'couch': 25,
'potted plant': 26,
'bed': 27,
'dining table': 28,
'toilet': 29,
'tv': 30,
'laptop': 31,
'sink': 32,
'refrigerator': 33}
 
id_to_label_map = {v: k for k, v in labels_dict.iteritems()}

file = open('label_map.pbtxt', 'w')

for key, value in id_to_label_map.iteritems():
    file.write('item {\n')
    file.write('\tid: ' + str(key)+'\n')
    file.write('\tname: \'' + value + '\'' + '\n')
    file.write('}\n')

file.close()    
