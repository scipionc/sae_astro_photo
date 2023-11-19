import cv2
import numpy as np

def remove_light_pollution(image_path):
    # Charge l'image
    image = cv2.imread(image_path)

    # Convertit l'image en espace de couleurs HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définit la plage de couleurs à conserver
    lower_bound = np.array([0, 0, 220]) # Rouge
    upper_bound = np.array([180, 255, 255]) # Jaune

    # Applique un masque pour conserver que les couleurs souhaitées
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Sauvegarde l'image
    cv2.imwrite('IMG/test/no_light_pollution.jpg', masked_image)

remove_light_pollution('IMG/barnard_stacked_gradient.png')