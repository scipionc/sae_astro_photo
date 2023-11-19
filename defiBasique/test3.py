import builtins
import tkinter as tk
import cv2
import numpy as np

def calculate_gradient_lineaire(image_path, gradient_lineaire_path):
    # Charge l'image et le gradient
    image = cv2.imread(image_path).astype(np.float64)
    gradient_lineaire = cv2.imread(gradient_lineaire_path).astype(np.float64)

    # Calcule le gradient
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Soustrait le gradient de l'image
    result = np.abs(image - gradient_lineaire).astype(np.uint8)

    # Retire les teintes jaunes
    hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    result[yellow_mask > 0] = [255, 255, 255]

    return result

def calculate_gradient_RBF(image_path, gradient_RBF_path):
    # Charge l'image et le gradient
    image = cv2.imread(image_path).astype(np.float64)
    gradient_RBF = cv2.imread(gradient_RBF_path).astype(np.float64)

    # Calcule le gradient
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Soustrait le gradient de l'image
    result = np.abs(image - gradient_RBF).astype(np.uint8)

    # Retire les teintes jaunes
    hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    result[yellow_mask > 0] = [255, 255, 255]

    return result

def load_image_lineaire():
    image_path = "IMG/barnard_stacked_gradient.png"
    gradient_lineaire_path = "IMG/gradient_lineaire.png"
    gradient_RBF_path = "IMG/gradient_lineaire.png"
    
    result = calculate_gradient_lineaire(image_path, gradient_lineaire_path)
    
    cv2.imshow("Image traitee", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def load_image_RBF():
    image_path = "IMG/barnard_stacked_gradient.png"
    gradient_lineaire_path = "IMG/gradient_lineaire.png"
    gradient_RBF_path = "IMG/gradient_lineaire.png"
    
    result = calculate_gradient_RBF(image_path, gradient_lineaire_path)
    
    cv2.imshow("Image traitee", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Traitement de la pollution lumineuse")
boutton_lineaire = tk.Button(root, text="Methode par gradient lineaire", command=load_image_lineaire)
boutton_lineaire.pack()
bouton_RBF = tk.Button(root, text="Methode par gradient RBF", command=load_image_RBF)
bouton_RBF.pack()
root.mainloop()
