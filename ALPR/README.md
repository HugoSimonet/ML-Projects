# ALPR - Automatic License Plate Recognition

Automatic license plate recognition system using scikit-learn and skimage for vehicle plate detection and character extraction.

## Overview

This project implements a three-stage pipeline for reading license plates from vehicle images:
1. License plate detection and localization
2. Character segmentation
3. Character recognition using SVC

## Algorithm

### Step 1: Detecting the License Plate

Convert car image to grayscale and binary (black/white pixels):
- Detected connected components on foreground
- Filtered components by size (remove noise)
- Filtered by position (plates near bottom of image)

Implementation: `localization.py`, `cca2.py`

### Step 2: Segmenting the Characters

Extract individual characters from the detected plate region:
- Applied CCA to find character-like components
- Resized each character to 20x20 pixels for recognition

Implementation: `segmentation.py`

### Step 3: Character Recognition

Train SVC classifier on normalized character images:
- Training data: 20x20 images of digits (0-9) and uppercase letters (A-Z)
- Validation: 4-fold cross-validation
- Predict each segmented character
- Output final license plate string

Implementation: `machine_train.py`, `prediction.py`

## Installation

```bash
pip install -r requirements.txt
```

Requirements: scikit-learn, scikit-image, numpy, opencv-python, pillow, matplotlib

## Usage

```python
# Step 1: Localize plate
python localization.py

# Step 2: Segment characters
python segmentation.py

# Step 3: Recognize characters
python prediction.py
```

## Results

Algorithm demonstration on test vehicle images:

![Plate Detection](results/Figure_1.png)

![Component Analysis](results/Figure_2.png)

![Character Segmentation](results/Figure_3.png)

See [RESULTS.md](RESULTS.md) for detailed pipeline analysis and performance metrics.

## Project Structure

```
ALPR/
├── localization.py      # Plate detection
├── cca2.py              # Connected component analysis
├── segmentation.py      # Character extraction
├── machine_train.py     # Train SVC model
├── prediction.py        # Character prediction
├── models/              # Saved SVC models
└── train/               # Training data
```

## License

MIT License - see LICENSE file for details.
