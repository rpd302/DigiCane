import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

category_dict = {'n13104059':  'tree',
                'n04402449':   'telephone pole',
                'n03074380':   'pillar',
                'n04549122':   'wall unit',
                'n03222176':   'door',
                'n02747177':   'dustbin',
                'n04314914':   'stairs'
		}

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text + '.JPEG',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     category_dict[member[0].text],
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ['train','test']:
        xml_df = xml_to_csv(directory)
        xml_df.to_csv(directory + '_labels.csv', index=None)
        print('Successfully converted xml to csv.')


main()
