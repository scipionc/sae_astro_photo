import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk

# Charge l'image et effectue les opérations de traitement
img = cv.imread('IMG/barnard_stacked_gradient.png', 0)
img = cv.GaussianBlur(img, (3, 3), 0)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
abs_grad_x = cv.convertScaleAbs(sobelx)
abs_grad_y = cv.convertScaleAbs(sobely)
modgrad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Variables globales
tk_images = {
    'img': None,
    'modgrad': None,
    'sobelx': None,
    'sobely': None
}

# Fonction pour afficher l'image dans une fenêtre Tkinter
def show_image(selected_image):
    root = tk.Tk()
    root.title('Image Processing with Tkinter')
    tk_image = tk_images[selected_image]
    label = tk.Label(root, image=tk_image)
    label.pack()
    root.mainloop()

# Fonction pour mettre à jour l'image affichée en fonction du bouton cliqué
def update_image(selected_image):
    show_image(selected_image)

# Créé des boutons pour sélectionner entre les images
root = tk.Tk()
root.title('Image Selection with Tkinter')

tk_images['img'] = ImageTk.PhotoImage(Image.fromarray(img))
tk_images['modgrad'] = ImageTk.PhotoImage(Image.fromarray(modgrad))
tk_images['sobelx'] = ImageTk.PhotoImage(Image.fromarray(sobelx))
tk_images['sobely'] = ImageTk.PhotoImage(Image.fromarray(sobely))

img_button = tk.Button(root, text='Original', command=lambda: update_image('img'))
img_button.pack(side='left')

modgrad_button = tk.Button(root, text='Module Gradient', command=lambda: update_image('modgrad'))
modgrad_button.pack(side='left')

sobelx_button = tk.Button(root, text='Sobel X', command=lambda: update_image('sobelx'))
sobelx_button.pack(side='left')

sobely_button = tk.Button(root, text='Sobel Y', command=lambda: update_image('sobely'))
sobely_button.pack(side='left')

root.mainloop()
