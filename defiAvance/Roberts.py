import cv2
import matplotlib.pyplot as plt
import numpy as np

def Roberts(img):
    try: 
        # Charge l'image
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise Exception("Failed to load the image.")

        # Défini les noyaux de convolution pour l'opérateur de Roberts
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])

        # Applique les noyaux de convolution pour calculer les gradients
        gradient_x = cv2.filter2D(image, -1, roberts_x)
        gradient_y = cv2.filter2D(image, -1, roberts_y)

        # Calcule le gradient de magnitude
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Mise à l'échelle manuelle de l'image en une image de 8 bits
        gradient_magnitude = (255 * (gradient_magnitude / np.max(gradient_magnitude))).astype(np.uint8)

        return image, gradient_x, gradient_y, gradient_magnitude

    except Exception as e:
        print("Error", e)
        return None

if __name__ == '__main__':
    img_path = 'IMG/input/barnard.png'
    filtered_image = Roberts(img_path)
    
    # Affiche les images filtrées
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(filtered_image[0], cmap='gray')
    axs[0].set_title('Image originale')
    axs[0].axis('off')
    axs[1].imshow(filtered_image[1], cmap='gray')
    axs[1].set_title('Gradient horizontal')
    axs[1].axis('off')
    axs[2].imshow(filtered_image[2], cmap='gray')
