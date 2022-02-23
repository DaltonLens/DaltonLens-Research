#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import time

from icecream import ic

from cvlog import cvlog

def generate_plot ():
    cv2.namedWindow ("plot", cv2.WINDOW_NORMAL)
    x = np.linspace(0, 1, 100)
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),dpi=100)
    ax.plot(x, np.sin(x*10.0), label="label1")

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow ("plot", image_from_plot)
    while cv2.waitKey (0) & 0xff != ord('q'):
        pass

if __name__ == "__main__":
    # generate_plot ()
    for i in range(0,2):
        print ("Parent still running")
        cvlog.image(np.random.rand(128,128,3), 'Test')
        time.sleep(1)
