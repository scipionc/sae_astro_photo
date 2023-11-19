import cv2
import os
import matplotlib.pyplot as plt
from Canny import *
from Roberts import *
from Sobel import *
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

input_directory = 'IMG/input/'
output_directory = 'IMG/output/'
os.makedirs(output_directory, exist_ok=True)

# Liste des fichiers d'images dans le répertoire d'entrée
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg','.png'))]

# Liste des filtres à appliquer
filters = [Sobel, Canny, Roberts]

fig = plt.figure(figsize=(15, 5*len(image_files)))
gs = gridspec.GridSpec(len(image_files), len(filters) + 2, width_ratios=[1, 1] + [2] * len(filters), wspace=0.1)

for row, image_file in enumerate(image_files):
    title = fig.add_subplot(gs[row, 0])
    title.text(0.5, 0.5, f"Image: {row + 1}", ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    title.axis('off')

    input_image_path = os.path.join(input_directory, image_file)
    image = cv2.imread(input_image_path)

    axs = [fig.add_subplot(gs[row, 1])]
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Image originale')
    axs[0].axis('off')
    
    for col, filter_func in enumerate(filters):
        filtered_image = filter_func(input_image_path)

        if filtered_image is not None:
            if isinstance(filtered_image, tuple):
                filtered_image = filtered_image[3]

            if filtered_image.dtype != 'uint8':
                filtered_image = cv2.convertScaleAbs(filtered_image)

            axs.append(fig.add_subplot(gs[row, col + 2]))
            axs[col + 1].imshow(filtered_image, cmap='gray')
            axs[col + 1].set_title(filter_func.__name__)
            axs[col + 1].axis('off')

            filter_name = filter_func.__name__.lower()
            output_image_path = os.path.join(output_directory, f"{filter_func.__name__}_{image_file}")
            cv2.imwrite(output_image_path, filtered_image)

plt.show()
