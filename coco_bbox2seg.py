"""
2023/09/05
This function will create mask values (segmentations) based on the values of bbox
Please note the area is also updated accordingly (area = width * height)

coco bbox format [xmin, ymin, width, height]
coco segmentation format [[xmin, ymin, xmin+width, ymin, xmin+width, ymin+height, xmin, ymin+height]]

x,y,w,h = anno['bbox']
anno['segmentation'] = [[x,y, x+w,y, x,y+h, x+w,y+h]]

"""

import json
import time

# float values
def bbox2seg(bbox: list):
    # input coco bbox values
    [xmin, ymin, width, height] = bbox
    # outoput segmentation values and bbox area
    return [[xmin, ymin, xmin+width, ymin, xmin+width, ymin+height, xmin, ymin+height]], width*height

'''
# integer values
def gen_int(val: float):
    return int(round(val, 0))
    
def bbox2seg(bbox: list):
    [xmin, ymin, width, height] = bbox
    xmin, ymin, width, height = gen_int(xmin), gen_int(ymin), gen_int(width), gen_int(height)
    return [[xmin, ymin, xmin+width, ymin, xmin+width, ymin+height, xmin, ymin+height]], width*height

'''

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

if __name__ == '__main__':
    inp = input('Type in the coco json path here >>>')

    start_time = time.perf_counter()
    ct = seg_gen(inp)
    print('run time (in seconds):', time.perf_counter() - start_time)
    print('target dir:\n', inp)
    print('processed annotations:', ct)
    print()