import cv2
import matplotlib.pyplot as plt

def Canny(img):
    try:
        # Charge l'image
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise Exception("Failed to load the image.")

        # Réduire la taille de l'image
        scale_percent = 70
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        smaller_image = cv2.resize(image, (width, height))

        # Applique l'opérateur de Canny pour détecter les contours
        edges = cv2.Canny(smaller_image, threshold1=100, threshold2=200)

        return edges
    
    except Exception as e:
        print("Error",e)
        return None

if __name__ == '__main__':
    img_path = 'IMG/input/barnard.png'
    filtered_image = Canny(img_path)
    
    # Affiche les images filtrées
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.show()
    