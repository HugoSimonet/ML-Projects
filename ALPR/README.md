# ALPR
## _Automatic License Plate Recognition_

Used skikit-learn and skimage to develop an algorithm that takes any image of a car, reads the characters on the license plate, and outputs a string

## Step 1: Detecting the License Plate
 - Convert the car image into a grayscale image then into a binary image so that every pixel is either black or white (See localization.py)
 (/ALPR/Figure_1.png)
 - Detected connected components on the foreground of the transformed image (See localization.py)
 - Filtered out the connected components that are too small to be considered a license plate, then the ones that aren't near the bottom of the image

 (/ALPR/Figure_2.png)

## Step 2: Segmenting the Characters

## Step 3: Character Recognition

