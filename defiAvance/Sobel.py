import cv2
import matplotlib.pyplot as plt

def Sobel(img):
    try: 
        # Charge l'image avec OpenCV
        image = cv2.imread(img)
        
        # Converti l'image en niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray_image is None:
            raise Exception("Failed to load the image.")
        
        # Applique le filtre de Sobel pour la détection des contours
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient horizontal
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient vertical

        # Combine les composantes X et Y pour obtenir le gradient total
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

        return gradient_magnitude
    
    except Exception as e:
        print("Error", e)
        return None
    
if __name__ == '__main__':
    img_path = 'IMG/input/barnard.png'
    filtered_image = Sobel(img_path)
    
    # Affiche l'image filtrée
    plt.imshow(filtered_image, cmap='gray')
    plt.show()
