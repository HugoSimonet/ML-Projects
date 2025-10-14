# ALPR
## _Automatic License Plate Recognition_\
Used skikit-learn and skimage to develop an algorithm that takes any image of a car, reads the characters on the license plate, and outputs a string\
## Step 1: Detecting the License Plate
 - Convert the car image into a grayscale image then into a binary image so that every pixel is either black or white (See localization.py)\
 ![](/ALPR/Figure_1.png)
 - Detected connected components on the foreground of the transformed image
 - Filtered out the connected components that are too small to be considered a license plate, then the ones that aren't near the bottom of the image (See cca2.py)\
 ![](/ALPR/Figure_2.png)\
## Step 2: Segmenting the Characters
 - Used CCA again to detect connected components on the license plate that resembled characters and stored those regions
 - Resized regions into 20x20 pixel regions for the next step, Character Recognition (See segmentation.py)\
 ![](/ALPR/Figure_3.png)\
## Step 3: Character Recognition
 - Used an SVC from skikit-learn to train a model based on 20x20 pixel images of numbers 0-9 and letters A-Z, capitals only, then stored that model (See /models, /train, and machine_train)
 - Model accuracy calculated using a 4-fold CV
 - Model used to predict each character in the original 20x20 pixel regions in our previous step
 - String of the license plate number is outputted
