# SAM-Assisted Image Labeling Tool

A Python-based labeling tool that uses [Segment Anything Model (SAM)](https://github.com/facebookresearch/sam2) to help you interactively segment objects in images. This tool provides a Tkinter front-end for easy setup and launching a Matplotlib-based interface for image segmentation. With it, you can:

1. Choose a directory of images.
2. Enter any number of class labels you wish to annotate.
3. Quickly segment objects by single clicks (or multiple clicks for one object).
4. Save annotations in a `.txt` format compatible with common YOLO-style tooling.
5. (Optionally) split your dataset into train/valid/test folders.

![Labeling Tool for YOLO](https://github.com/user-attachments/assets/45e268f5-b0d6-44b4-b591-d39ed6dc44dc)

## Features

- **Tkinter Front-End:** Simple interface to choose your images folder and input labels.
- **Segment Anything Model (SAM):** Uses the `ultralytics` library’s `SAM` implementation. 
- **Interactive Labeling:** Click on a point in the image to propose a mask; right-click to remove an object.
- **Annotation Saving:** Saves YOLO-style annotations (`class_id`, bounding box, plus full polygon).
- **Dataset Splitting:** Includes an optional utility to split data into `train`, `valid`, and `test` subfolders, complete with a `data.yaml` file.

---

## Requirements and Dependencies

1. **Python 3.7+** (tested up to 3.11).
2. [OpenCV](https://pypi.org/project/opencv-python/) (`pip install opencv-python`)
3. [numpy](https://pypi.org/project/numpy/) (`pip install numpy`)
4. [matplotlib](https://pypi.org/project/matplotlib/) (`pip install matplotlib`)
5. [tkinter](https://docs.python.org/3/library/tkinter.html) (comes standard with most Python installations, but please ensure it’s installed and enabled).
6. [ultralytics](https://pypi.org/project/ultralytics/) for SAM (`pip install ultralytics` or `pip install ultralytics==<specific_version>`)


## Recommended Installation Method for Beginners
1. **Install [Anaconda](https://docs.anaconda.com/anaconda/install/).** During installation, I recommend checking the option to include Anaconda to your PATH environment variable.
2. **Create a Python environment with Python version 3.9:**  On the Command Prompt, simply type:  `conda create -n project_environment_name python=3.9`  
   Please be sure to change "project_environment_name to the name of the environment you wish to use.
3. **Activate the environment:**    On the command prompt, type:   `conda activate project_environment_name`       
   (again, use the name of the environment you have chosen)  
4. **Install Ultralytics:** On the command prompt, type `pip install ultralytics`
5. **Optional GPU-Based Ultralytics Installation:**  If you have an NVIDIA GPU, you will be able to utilize it for much faster processing for labeling and training. The following command should install everything neeeded (as of Dec. 2024): (`conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics`  )    

Additionally, you will need a SAM weights file named (by default) **`sam2.1_b.pt`** in the same directory as the script—or you can modify the code to point to your own model file. When running the code for the first time, the **`sam2.1_b.pt`** file will be downloaded into the directory automatically.

---
