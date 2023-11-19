import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('IMG/input/pollution2.png')

# Applique un flou gaussien à l'image pour réduire le bruit
img = cv.GaussianBlur(img, (3, 3), 0)

# Calcule le gradient horizontal en utilisant l'opérateur de Sobel
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)

# Calcule le gradient vertical en utilisant l'opérateur de Sobel
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# Convertit les valeurs de gradient en valeurs absolues
abs_grad_x = cv.convertScaleAbs(sobelx)
abs_grad_y = cv.convertScaleAbs(sobely)

# Calcule le module du gradient en combinant les gradients X et Y
modgrad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Affiche les images dans une disposition en 2x2 à l'aide de Matplotlib
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(modgrad, cmap='gray')
plt.title('Module gradient'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()