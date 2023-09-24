# Document Layout Analysis from Data Labeling to Model Retraining
- keyword: ```computer vision```, ```object detection```,  ```document layout analysis```,  ```Vision Transformer (ViT)```, ```Document image Transformer (DiT)```, ```fine-tune```, ```retrain```, ```PyTorch```, ```detectron2```
- highlight:
  - ```label conversion from labelme bboxes to coco masks```
  - ```retrain model pretrained for multi-classes object detection into single-class```

<p align="center"><img src="https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/example_coco_masks.png" style = "border:10px solid white"></p> 
<p align="center"><img src="https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/example_labelme_bbox.png" style = "border:10px solid white"></p> 


## 1 Custom data labeling and preprocess
### Note1: common datasets for document layout
- DocLayNet (2023) 28GB, 80863 pages {financial_reports,scientific_articles,laws_and_regulations, government_tenders, manuals, patents}
- DocBank (2020) 50GB, 500K pages from arXiv.com {scientifi articles}
- PubLayNet (2019) 96GB, over 1 million pdf from PubMed {scientific articles}
- RVL-CDIP (2015) 37GB, 400,000 grayscale images, 16 classes
  - {"letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"}
- PRImA Layout Analysis Dataset (2009) 1240 pages {magazines pages, technical articles}

### Note2: labeling tools
- labelme  https://github.com/wkentaro/labelme   
  - very basic functions and app can crush sometimes (remember to enable autosave)
  - outut label json file for each image separately 
  - can use third party tools to convert labelme output to cooc format: such as https://github.com/fcakyon/labelme2coco
- label studio https://labelstud.io/
  - mature but with decent learning curve
  - can natively output label in coco format (therefore, label data of all the images are compiled into one json file)

### LabelMe label output format
- example json file below shows three different labels in both rectangle (bbox) and polygon (mask) formats
- point coordinates for rectangle ```[[x1, y1], [x3, y3]]```
- point coordinates for polygon (mask) ```[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]```
```
{"imagePath": "123.jpg",
  "imageData": null,
  "shapes":
      [
      {"shape_type": "rectangle", "points": [[204.9180327868852, 131.14754098360655], [1529.5081967213114, 1944.2622950819673]], "flags": {}, "group_id": null, "label": "section"},
      {"shape_type": "rectangle", "points": [[1585.245901639344, 78.68852459016394], [200.97022415523588, 1.4210854715202004e-14]], "flags": {}, "group_id": null, "label": "Header"},
      {"shape_type": "rectangle", "points": [[204.9180327868852, 1957.377049180328], [537.7049180327868, 2054.0983606557375]], "flags": {}, "group_id": null, "label": "section"},
      {"shape_type": "polygon", "points": [[204.9180327868852, 131.14754098360655], [204.9180327868852, 1944.2622950819673], [1529.5081967213114, 1944.2622950819673], [1529.5081967213114, 131.14754098360655]], "flags": {}, "group_id": null, "label": "section"},
      {"shape_type": "polygon", "points": [[1585.245901639344, 78.68852459016394], [1585.245901639344, 1.4210854715202004e-14], [200.97022415523588, 1.4210854715202004e-14], [200.97022415523588, 78.68852459016394]], "flags": {}, "group_id": null, "label": "Header"},
      {"shape_type": "polygon", "points": [[204.9180327868852, 1957.377049180328], [204.9180327868852, 2054.0983606557375], [537.7049180327868, 2054.0983606557375], [537.7049180327868, 1957.377049180328]], "flags": {}, "group_id": null, "label": "section"}
      ],
  "version": "5.1.1",
  "flags": {},
  "imageHeight": 2200,
  "imageWidth": 1700}
```
## 2 Setup environment
- below is the version of PyTorch and detectron2 I installed and ran successfully on both AWS EC2 g4dn.2xlarge instance (NVIDIA Turing T4 16GB VRAM) and a local workstation (NVIDIA Turing TU104 8GB VRAM)
```
torch:  2.0 ; cuda:  cu118
detectron2: 0.6
```
- Install cuda and then PyTorch
- Build detectron2 from source ```python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' ```
- Clone unilm repo (DiT is part of unilm) ```git clone https://github.com/microsoft/unilm.git```
  - If you use Python3.10 or above, you will encounter this error ```ImportError: cannot import name 'Iterable' from 'collections'```
  - Please modify *unilm/dit/object_detection/ditod/table_evaluation/data_structure.py* replace line 6 as ```Iterable from collections.abc```
  - for more details, please check **DiT_tutorial.ipynb**
## 3 Model retraining
- please refer to [DiT_tutorial.ipynb](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/DiT_tutorial.ipynb)
