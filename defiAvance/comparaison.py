import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Canny import Canny
from Roberts import Roberts
from Sobel import Sobel

# Chemin de l'image
img_path = 'IMG/input/barnard.png'

# Nombre d'itérations pour mesurer le temps d'exécution
num_iterations = 100

# Mesure du temps d'exécution de la méthode Canny
canny_times = []
for _ in range(num_iterations):
    start_time = time.time()
    Canny(img_path)
    end_time = time.time()
    canny_times.append(end_time - start_time)

# Mesure du temps d'exécution de la méthode Roberts
roberts_times = []
for _ in range(num_iterations):
    start_time = time.time()
    Roberts(img_path)
    end_time = time.time()
    roberts_times.append(end_time - start_time)

# Mesure du temps d'exécution de la méthode Sobel
sobel_times = []
for _ in range(num_iterations):
    start_time = time.time()
    Sobel(img_path)
    end_time = time.time()
    sobel_times.append(end_time - start_time)

# Affichage des temps d'exécution moyens
print(f"Temps d'exécution moyen pour Canny : {np.mean(canny_times):.5f} secondes")
print(f"Temps d'exécution moyen pour Roberts : {np.mean(roberts_times):.5f} secondes")
print(f"Temps d'exécution moyen pour Sobel : {np.mean(sobel_times):.5f} secondes")

# Tracé d'un diagramme en barres pour comparer les temps d'exécution
methods = ['Canny', 'Roberts', 'Sobel']
execution_times = [np.mean(canny_times), np.mean(roberts_times), np.mean(sobel_times)]

plt.bar(methods, execution_times)
plt.ylabel('Temps d\'exécution moyen (secondes)')
plt.title('Comparaison des temps d\'exécution')
plt.show()
