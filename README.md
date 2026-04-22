# Water Meter YOLO Labeler

A simple tool for labeling water meter images in YOLO format.

## What this tool does

This tool helps you:

- open a folder of images
- draw bounding boxes
- assign the correct class to each box
- move and resize boxes
- rotate images before labeling
- save labels in YOLO `.txt` format

---

## Supported classes

The tool uses the following classes:

- `meter`
- `window`
- `0`
- `1`
- `2`
- `3`
- `4`
- `5`
- `6`
- `7`
- `8`
- `9`
- `unknown`

---

## How to run the tool

Make sure Python and the required packages are installed.

### Install dependencies

```bash
pip install opencv-python PySide6