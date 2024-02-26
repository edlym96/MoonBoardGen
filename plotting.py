import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
import os
import numpy as np

IMG_X_START = 97
IMG_Y_START = 88
IMG_DX = IMG_DY = 51.3
CIRCLE_SIZE = 23

COLOR_MAP = [
        (0, "#75fc4a"), # green
        (1, "#0500ea"), # red
        (2, "#e03321")  # blue
    ]

MOONBOARD_IMG = "./images/mbsetup-2016-min.jpeg"

def plot_board_2016(coords):
    """
    Plots a moonboard problem onto a board
    """
    img = plt.imread(MOONBOARD_IMG)

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    ax.imshow(img)
    for idx, color in COLOR_MAP:
        holds = np.argwhere(coords[idx])
        for hold_row, hold_col in holds:
            circ = Circle((IMG_X_START + hold_col * IMG_DX, IMG_Y_START + hold_row * IMG_DY), CIRCLE_SIZE, fill=False, color=color, linewidth=1.35)
            ax.add_patch(circ)

    plt.show()