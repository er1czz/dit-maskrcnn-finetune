# Document Layout Analysis from Data Labeling to Model Retraining
keyword ```computer vision```, ```document layout analysis```, ```transformer```, ```mask rcnn```

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
  - app can ocassionally crush with limited functions 
  - can use third party tools to convert labelme output to cooc format: such as https://github.com/fcakyon/labelme2coco
- label studio https://labelstud.io/
  - mature but with decent learning curve
  - natively convert output to coco format

## 2 Model retraining
- 
