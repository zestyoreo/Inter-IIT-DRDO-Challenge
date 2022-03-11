#Pencil Edge Figure
import cv2
import numpy as np
import matplotlib.pyplot as plt

for name in ['inter/1.png','inter/2.png','inter/3.png','inter/4.png','inter/5.png']:
    img = cv2.imread(name,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img,100,200)

    plt.subplot(131),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(edges)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Grey Image'), plt.xticks([]), plt.yticks([])
    plt.show()
