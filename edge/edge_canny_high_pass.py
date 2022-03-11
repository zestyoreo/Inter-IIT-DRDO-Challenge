#Pencil Edge Figure
import cv2
import numpy as np
import matplotlib.pyplot as plt

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

i=0
for name in ['inter/1.png','inter/2.png','inter/3.png','inter/4.png','inter/5.png']:
    img = cv2.imread(name,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img,100,200)

    high_pass = highpass(img, 3)
    edge_pass = highpass(edges, 3)
    i+=1
    plt.subplot(131),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Grey Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(edge_pass, cmap = 'gray')
    plt.title('Edge grey high pass'), plt.xticks([]), plt.yticks([])
    #plt.savefig(str(i)+'_edge.png')
    plt.show()
