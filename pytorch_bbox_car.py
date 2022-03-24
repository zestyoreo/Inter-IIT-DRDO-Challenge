import os
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
import math
from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
import time
import cv2

def yolo():
    board=Darknet(config_file_path,inference=True)
    board.load_weights(weight_file_path)
    board.cuda()
    return board


def my_detect(m,cv_img):
    use_cuda=True
    img=cv2.resize(cv_img, (m.width, m.height))
    # print(img.shape)
    # print(img)
    # img = np.array(img, dtype=np.int16)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, img, 0.2, 0.6, use_cuda)
    if len(boxes[0])==0:
        return [False,0,0,0,0]
    box=boxes[0][0]
    h,w,c=cv_img.shape
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    return [True,x1,y1,x2,y2]

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

def depth_data_to_edge_pass():
    b = np.load("depth_file_name")          
    b = b-np.min(b)
    b = b/np.max(b)
    b = 255*b
    b = np.floor(b)

    aperture_size = 5
    img = np.uint8(b)
    edges = cv2.Canny(img,1,100,apertureSize=aperture_size)

    high_pass = highpass(img, 3)
    edge_pass = highpass(edges, 3)  #numpy array
    plt.imshow(edge_pass, cmap = 'gray')
    plt.savefig('depth_edge.png')
    
def get_bbox_coords():          #returns cooordinates of and saves image inside of bbox if there
    board = yolo()
    depth_data_to_edge_pass()
    yolo_inp_img = cv2.imread('depth_edge.png')     #read edge_pass black&white image
    yolo_inp_img = yolo_inp_img[33:252, 72:366]     #crop

    frame = yolo_inp_img
    ret,x1,y1,x2,y2 = my_detect(board,frame)        #bounding box

    #showing bb for verification purposes
    if ret :
      frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
      #crop and save bb
      car_img = yolo_inp_img[x1:x2, y1:y2] 
      cv2.imwrite("car_cropped.png", car_img)
    plt.imshow(frame),plt.show()

    return ret,x1,y1,x2,y2      #ret is whether bbox is present or not