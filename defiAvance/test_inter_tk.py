import cv2
import os
import tkinter as tk
from Canny import *
from Roberts import *
from Sobel import *
from PIL import Image, ImageTk

root = tk.Tk()

input_directory = 'IMG/input/'

output_directory = 'IMG/output/'

os.makedirs(output_directory, exist_ok=True)

# Liste des fichiers d'images dans le répertoire d'entrée
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png'))]

# Liste des filtres à appliquer
filters = [Sobel, Canny, Roberts]

def display_image(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=img)
    label.image = img
    label.pack()

for image_file in image_files:
    input_image_path = os.path.join(input_directory, image_file)
    image = cv2.imread(input_image_path)
    
    window = tk.Toplevel(root)
    window.title(image_file)
    
    display_image(image)

    for filter_func in filters:
        filtered_image = filter_func(image)
        if filtered_image is not None:
            display_image(filtered_image)

root.mainloop()
