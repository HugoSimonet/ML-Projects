from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

label_image = measure.label(localization.binary_car_image)
fig, (ax1) = plt.subplots(1)
ax1.imshow(localization.gray_car_image, cmap="gray")

for region in regionprops(label_image):
    if region.area < 50:
        continue

    min_row, min_col, max_row, max_col = region.bbox
    rect_border = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rect_border)

plt.show()