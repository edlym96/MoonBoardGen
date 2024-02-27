import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
import os
import numpy as np

IMG_X_START = 97
IMG_Y_START = 88
IMG_DX = IMG_DY = 51.3
CIRCLE_SIZE = 23
TEXT_OFFSET_X = 12.5
TEXT_OFFSET_Y = 18

COLOR_MAP = {
        0: "#75fc4a", # green
        1: "#0500ea", # red
        2: "#e03321"  # blue
}

MOONBOARD_IMG = "./images/mbsetup-2016-min.jpeg"

def plot_board_helper(board, fig, ax, ax_slider, img, init_threshold=0.2):
    # board[1] is array of move holds
    text_holds = np.argwhere(board[1] >= 0.05)

    start_rows, start_cols = np.unravel_index(np.argsort(board[0], axis=None)[::-1][:2], board[0].shape)
    end_rows, end_cols = np.unravel_index(np.argsort(board[2], axis=None)[::-1][:2], board[2].shape)

    # Calculate if there's a possible second start and end hold up front 
    is_second_start = board[0, start_rows[1], start_cols[1]] > 0.1
    is_second_end = board[2, end_rows[1], end_cols[1]] > 0.1

    def plot_circle(ax, hold_row, hold_col, color, **kwargs):
        circ = Circle((IMG_X_START + hold_col * IMG_DX, IMG_Y_START + hold_row * IMG_DY), CIRCLE_SIZE, fill=False, color=color, linewidth=1.35, **kwargs)
        ax.add_patch(circ)

    def plot_text(ax, hold_row, hold_col, z_idx, **kwargs):
        ax.text(IMG_X_START + hold_col * IMG_DX + TEXT_OFFSET_X, IMG_Y_START + hold_row * IMG_DY + TEXT_OFFSET_Y, f"{board[z_idx][hold_row][hold_col]:.2f}", fontsize=6, fontstretch=1000, **kwargs)

    def update_holds(threshold):
        ax.cla()
        ax.imshow(img)
        # Plot start
        plot_circle(ax, start_rows[0], start_cols[0], COLOR_MAP[0])
        plot_text(ax, start_rows[0], start_cols[0], 0)
        if is_second_start:
            plot_circle(ax, start_rows[1], start_cols[1], COLOR_MAP[0], alpha=0.3)
            plot_text(ax, start_rows[1], start_cols[1], 0)

        # Plot holds
        holds = np.argwhere(board[1] > threshold)
        for hold_row, hold_col in holds:
            plot_circle(ax, hold_row, hold_col, COLOR_MAP[1])
        for hold_row, hold_col in text_holds:
            plot_text(ax, hold_row, hold_col, 1)

        # Plot end
        plot_circle(ax, end_rows[0], end_cols[0], COLOR_MAP[2])
        plot_text(ax, end_rows[0], end_cols[0], 2)
        if is_second_end:
            plot_circle(ax, end_rows[1], end_cols[1], COLOR_MAP[2], alpha=0.3)
            plot_text(ax, end_rows[1], end_cols[1], 2)

        fig.canvas.draw_idle()

    # Initialise hold plot
    update_holds(init_threshold)

    # Add slider
    sliderwave = Slider(ax_slider, 'Hold Threshold', 0.05, 1, valinit=init_threshold)
    sliderwave.on_changed(update_holds)

    # Need to return slider object here to provider reference to original object (prevents slider from being unresponsive)
    return sliderwave


def plot_board_2016(board, init_threshold=0.2):
    """
    Plots a moonboard problem onto a board
    """
    img = plt.imread(MOONBOARD_IMG)

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1, figsize=(10,9))
    ax.set_aspect('equal')
    axwave = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    # keep reference to slider object
    slider = plot_board_helper(board, fig, ax, axwave, img, init_threshold)
    
    plt.show()