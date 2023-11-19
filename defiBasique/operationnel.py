import cv2
import numpy as np
from matplotlib import pyplot as plt

def img_sans_pollution():
    # Chemin d'accès aux images d'entrée
    image_path = "IMG/input/barnard.png"
    gradient_lineaire_path = "IMG/gradient_lineaire.png"
    gradient_RBF_path = "IMG/gradient_lineaire.png"
    
    # Charge l'image et les gradients à partir des fichiers d'image
    image = cv2.imread(image_path).astype(np.float64)
    gradient_lineaire = cv2.imread(gradient_lineaire_path).astype(np.float64)
    gradient_RBF = cv2.imread(gradient_RBF_path).astype(np.float64)
    
    # Calcule la différence entre l'image et les gradients
    result = np.abs(image - gradient_lineaire - gradient_RBF).astype(np.uint8)
    
    # Affiche l'image en utilisant Matplotlib
    plt.figure()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Image traitée")
    plt.axis('off')
    plt.show()

img_sans_pollution()
