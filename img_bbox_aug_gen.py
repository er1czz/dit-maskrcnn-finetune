'''
This script is to generate a new dataset with augmentation
Enter the full path of dataset including images and labelme json files.
Output with "_AuG" inserted in the names of a new directory and individual files.
By Eric in Jan 2024

Next: parallelization
'''

import albumentations as A
import cv2
import json, os, time
from matplotlib import pyplot as plt

def gen_bboxes(img_path, lab_path):
    # input labelme json label path
    # output list of bbox in a format as follows
    # [x_min, y_min, w, h, label_name]
    image = cv2.imread(img_path)
    hmax, wmax = image.shape[0], image.shape[1]
    
    # Opening JSON file
    f = open(lab_path)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()

    # create list of bboxes
    bboxes = []
    # convert bbox labelme to min max
    for i in range(len(data['shapes'])):
        [[x0, y0], [x1, y1]] = data['shapes'][i]['points']
        x_min, x_max = max(min(x0, x1),0), min(max(x0, x1), wmax)
        y_min, y_max = max(min(y0, y1),0), min(max(y0, y1), hmax)
        w, h = (x_max - x_min), (y_max - y_min)
        label = data['shapes'][i]['label']
        bbox = [x_min, y_min, w, h]
        bbox = [round(i, 2) for i in bbox]
        bbox.append(label)
        bboxes.append(bbox)

    return bboxes
    
def main(dir_inp):
    inp_path, f = os.path.split(dir_inp)
    dir_sav = inp_path + '_AuG'
    # Create a new directory because it does not exist
    isExist = os.path.exists(dir_sav)
    if not isExist:
        os.makedirs(dir_sav)
        print('created new dir:', dir_sav)
        
    json_files = [file for file in os.listdir(dir_inp) if file.endswith('.json')]
    json_files.sort()
    ext = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.tiff', '.TIFF', '.bmp', '.BMP']
    imag_files = [file for file in os.listdir(dir_inp) if file.endswith(tuple(ext))]
    imag_files.sort()

    # define transform function
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='coco'))
    
    # counter
    ct = 0
    for img in imag_files:
        name0 = os.path.splitext(img)[0]
        ext = os.path.splitext(img)[1]
        
        img_path = os.path.join(dir_inp, img)
        lab_path = os.path.join(dir_inp, name0 +'.json')
    
        name_A = name0 + '_AuG'
        img_path_A = os.path.join(dir_sav, name_A+ext)
        lab_path_A = os.path.join(dir_sav, name_A+'.json')

        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
        # load labelme json file
        f = open(lab_path)
        data = json.load(f)
        f.close()
        
        bboxes = gen_bboxes(img_path, lab_path)
        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
    
        # save augmented image
        plt.imsave(img_path_A, transformed_image)
        print('Processed image:', name0+ext, '>>>> Saved as:', name_A+ext)
        
        # copy json dictionary 
        g = data
        
        # loop and update bbox
        for i in range(len(transformed_bboxes)):
            box = transformed_bboxes[i]
            g["shapes"][i]["label"] = box[-1]
            [x_min, y_min, w, h] = box[:-1]
            x0, y0, x1, y1 = int(x_min), int(y_min+h), int(x_min+w), int(y_min)
            g["shapes"][i]["points"] = [[x0, y0], [x1, y1]]
            g["shapes"][i]["group_id"]=None 
            g["shapes"][i]["shape_type"]="rectangle" 
            g["shapes"][i]["flags"]={} 
        g["imagePath"]=name_A+ext
        g["imageHeight"]=transformed_image.shape[0]
        g["imageWidth"]=transformed_image.shape[1] 
        # save augmented label as json
        with open(lab_path_A, "w") as outfile:
            json.dump(g, outfile, indent=4)
        print('Processed label:', name0+'.json', '>>>> Saved as:', name_A+'.json')    
        ct += 1
    return ct, dir_sav
    
if __name__ == '__main__':
    dir_inp = input('Enter full path of input dataset here (labelme json + image) >>>')
    start_time = time.perf_counter()
    ct, dir_sav = main(dir_inp)

    print('run time (in seconds):', time.perf_counter() - start_time)
    print('target dir:\n', dir_inp)
    print('saved dir:\n', dir_sav)
    print('In total processed:', ct, 'units!')
    print()
