import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/50_whorl.jpg", cv2.IMREAD_GRAYSCALE)
edges_1 = cv2.Canny(img, 100, 200)
edges_2 = cv2.Canny(img, 200, 300)
