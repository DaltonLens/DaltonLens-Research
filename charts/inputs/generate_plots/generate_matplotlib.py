#!/usr/bin/env python3

from dlcharts.common.cvlog import cvlog

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import time

from icecream import ic

def generate_plot ():
    x = np.linspace(0, 1, 10)    
        
    for i in range(1, 10):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),dpi=100)
        ax.plot(x, np.sin(x*i), label="label1")
        fig.canvas.draw()
        cvlog.plot (fig)
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cvlog.image (image_from_plot)
        time.sleep (1)

if __name__ == "__main__":

    # Should be started before creating any figure.
    cvlog.enabled = True

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),dpi=100)
    plt.close(fig)

    plt.ioff()
    generate_plot ()

    cvlog.waitUntilWindowsAreClosed()

    # cv2.imshow ('Test Image', np.random.rand(128,128,3))
    # while True:
    #     cv2.waitKey (0)
