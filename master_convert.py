This script will process labelme labels (plural: json files) and output coco label (single json file).
1) First step: generate coco label file from labelme label files via labelme2coco library
2) Second step: generate masks (segmentation) from bboxes with custom functions: bbox2seg and seg_gen

Reference:

coco bbox format [xmin, ymin, width, height]
coco segmentation format [[xmin, ymin, xmin+width, ymin, xmin+width, ymin+height, xmin, ymin+height]]

x,y,w,h = anno['bbox']
anno['segmentation'] = [[x,y, x+w,y, x,y+h, x+w,y+h]]

https://github.com/fcakyon/labelme2coco/blob/master/labelme2coco/__init__.py
'''
import json
import time

from sahi.utils.file import save_json
from labelme2coco.labelme2coco import get_coco_from_labelme_folder
from pathlib import Path

def bbox2seg(bbox: list):
    # input coco bbox values
    [xmin, ymin, width, height] = bbox
    # outoput segmentation values and bbox area
    return [[xmin, ymin, xmin+width, ymin, xmin+width, ymin+height, xmin, ymin+height]], width*height

def seg_gen(json_file_path):
    # read coco json file
    coco_data = json.load(open(json_file_path))
    # how many annotations will be revised
    ct = 0
    for i in coco_data['annotations']:
        i['segmentation'], i['area'] = bbox2seg(i['bbox'])
        ct += 1
    # save updated json file and overwrite existing one
    with open(json_file_path, 'w') as f:
        json.dump(coco_data, f)
    f.close()
    return ct

def main(_labelme_folder:str, _output_name:str):
    coco_file = get_coco_from_labelme_folder(labelme_folder=_labelme_folder, skip_labels=[])
    save_path = str(Path(_labelme_folder) / (_output_name + ".json"))
    save_json(coco_file.json, save_path)
    return save_path


if __name__ == "__main__":
    labelme_folder = input('Enter the full path of labelme dir: ')
    output_name = input('Enter the output coco json file name (without .json): ')

    start_time = time.perf_counter()

    # convert labelme to coco
    try: 
        save_path = main(labelme_folder, output_name)
    except:
        raise Exception('If you encouter this error, there could be a coco file already exists.')

    # generate masks from bbox
    ct = seg_gen(save_path)
    print()
    print('Generated masks (from bboxes):', ct)
    print('coco label file saved to:\n', save_path)
    print('Job Done (in seconds):', time.perf_counter() - start_time)
