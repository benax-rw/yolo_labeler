Water Meter YOLO Labeler

A graphical labeling tool for preparing water meter datasets in YOLO format.

This project is part of the Rwanda Water Meter Reading System, an AI-based computer vision project for automatic water meter detection and reading.

⸻

Features

The tool allows users to:

* open a folder of water meter images
* draw YOLO bounding boxes
* assign classes to objects
* select and edit boxes
* move bounding boxes
* resize bounding boxes
* rotate images before labeling
* remove poor images
* save labels in YOLO .txt format
* prepare datasets for YOLO training

⸻

Supported Classes

The labeling tool supports the following classes:

Class ID	Class Name	Description
0	meter	Full physical water meter
1	window	Reading/display window
2	0	Digit 0
3	1	Digit 1
4	2	Digit 2
5	3	Digit 3
6	4	Digit 4
7	5	Digit 5
8	6	Digit 6
9	7	Digit 7
10	8	Digit 8
11	9	Digit 9
12	unknown	Unclear/unreadable digit

⸻

Dataset Structure

Expected dataset structure:

raw_dataset/
├── images/
└── labels/

Example:

raw_dataset/
├── images/
│   ├── wm_0001.jpg
│   ├── wm_0002.jpg
│   └── ...
│
└── labels/
    ├── wm_0001.txt
    ├── wm_0002.txt
    └── ...

⸻

YOLO Label Format

Each line inside a .txt file follows:

class_id x_center y_center width height

Example:

2 0.309783 0.450836 0.028986 0.030898

Meaning:

class 2 → digit 0

Coordinates are normalized between 0 and 1.

⸻

Installation

Install dependencies

pip install opencv-python PySide6

⸻

How to Run

Run:

python3 01_label_images.py

Then:

1. Select the image folder
2. Start drawing bounding boxes
3. Assign the correct class
4. Save labels

⸻

Labeling Guidelines

Meter Bounding Box

* Cover the full physical water meter
* Avoid excessive background

Window Bounding Box

* Cover only the reading window
* Do not include:
    * screws
    * circular dials
    * empty plastic
    * background

Digit Bounding Boxes

* Keep boxes tight around digits
* Avoid excessive empty space

Unknown Class

Use unknown only when the digit is:

* blurry
* partially hidden
* unreadable
* ambiguous

If labeling a digit may confuse the model, it is better not to label it than to force an incorrect label.

⸻

Important Notes

* Tight bounding boxes improve accuracy
* Poor labels reduce model quality
* More diverse images improve generalization
* Images with minimal rotation usually give better results
* Strong rotation may cause digit confusion
* Remove very blurry or unusable images

⸻

Training Pipeline

Typical workflow:

Data Collection
 ↓
Data Labeling
 ↓
Dataset Preparation
 ↓
YOLO Training
 ↓
Retraining
 ↓
Prediction
 ↓
Reading Reconstruction

⸻

Related Scripts

Script	Purpose
01_label_images.py	Label water meter images
02_prepare_yolo_dataset.py	Prepare and validate YOLO dataset
03_train_yolo.py	Train YOLO model
04_predict_on_test_images.py	Run prediction on test images
05_retrain.py	Continue training from latest best model

⸻

Future Work

Planned improvements:

* segmentation-based reading
* webcam live inference
* mobile deployment
* real-time reading validation
* MQTT/cloud integration
* automatic reading reconstruction

⸻

Author

Gabriel BAZIRAMWABO
Rwanda Coding Academy | Benax Technologies
May 2026
