import pygame
import numpy as np
import torch
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame_widgets.button import Button
import enum
import asyncio

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
from constants import GRADE_MAP_REV, N_GRADES

from model import MODEL_PATH, MoonBoardCVAE, generate_board

# Initialize pygame
pygame.init()

# Define constants for the screen width and height
SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960

# Create the screen object
# The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class HoldColor(enum.Enum):
        START=0,
        MOVE=1,
        END=2,

class Board(pygame.sprite.Sprite):

    COLOR_MAP = {
        HoldColor.START: (117, 252, 74), # green
        HoldColor.MOVE: (5, 0, 234), # blue
        HoldColor.END: (224, 51, 33) #red
    }

    IMG_DY = 40
    IMG_DX = 41.3

    TEXT_OFFSET_X = 5
    TEXT_OFFSET_Y = 10
    DEFAULT_THRESHOLD = 0.2

    def __init__(self, model_obj, threshold=DEFAULT_THRESHOLD):
        super(Board, self).__init__()
        surf = pygame.image.load("./images/mbsetup-2016-min.jpeg").convert()
        self.surf = pygame.transform.smoothscale(surf, (SCREEN_WIDTH, SCREEN_HEIGHT - 160))
        # self.surf.set_colorkey((0, 0, 0), RLEACCEL)
        self.rect = self.surf.get_rect()
        self.hold_surf = pygame.Surface([self.rect.width, self.rect.height], pygame.SRCALPHA)
        self.hold_surd = self.hold_surf.convert_alpha()
        self.model = model_obj
        # Init board at 6B+ grade
        self.grade = '6B+'
        self.generate_new_board()

        self.threshold = threshold
        self.font = pygame.font.SysFont(None, size=15)
    
    def generate_new_board(self):
        print(f"Generating new board with grade {self.grade}")
        board_mat = generate_board(model_obj, self.grade)
        self.board_mat = board_mat
        # board[1] is array of move holds
        self.text_holds = np.argwhere(board_mat[1] >= 0.05)

        # self.board_mat = board_mat
        self.start_rows, self.start_cols = np.unravel_index(np.argsort(board_mat[0], axis=None)[::-1][:2], board_mat[0].shape)
        self.end_rows, self.end_cols = np.unravel_index(np.argsort(board_mat[2], axis=None)[::-1][:2], board_mat[2].shape)

        # Calculate if there's a possible second start and end hold up front 
        self.is_second_start = board_mat[0, self.start_rows[1], self.start_cols[1]] > 0.1
        self.is_second_end = board_mat[2, self.end_rows[1], self.end_cols[1]] > 0.1

        # redraw holds
        self.threshold = self.DEFAULT_THRESHOLD
        self._redraw_holds()

    def draw_holds(self):
        # Plot start
        self._draw_circle(self.start_rows[0], self.start_cols[0], HoldColor.START)
        if self.is_second_start:
            self._draw_circle(self.start_rows[1], self.start_cols[1], HoldColor.START, alpha=80)

        # Plot end∂
        self._draw_circle(self.end_rows[0], self.end_cols[0], HoldColor.END)
        if self.is_second_end:
            self._draw_circle(self.end_rows[1], self.end_cols[1], HoldColor.END, alpha=80)        
        
        # Plot holds
        holds = np.argwhere(self.board_mat[1] > self.threshold)
        for hold_row, hold_col in holds:
            self._draw_circle(hold_row, hold_col, HoldColor.MOVE)
    
    def set_grade(self, grade):
        self.grade = grade

    def _draw_circle(self, row, col, color: HoldColor, alpha=None):
        color = pygame.Color(*self.COLOR_MAP[color], 255 if alpha is None else alpha)
        pygame.draw.circle(self.hold_surf, color, (80 + col * self.IMG_DX, 70 + row * self.IMG_DY), 20, 2)
    
    def _draw_hold_text_helper(self, row, col, z_idx, screen):
        val = f"{self.board_mat[z_idx][row][col]:.2f}"
        text = self.font.render(val, True, pygame.Color(0, 0, 0))
        screen.blit(text, (80 + col * self.IMG_DX + self.TEXT_OFFSET_X, 70 + row * self.IMG_DY + self.TEXT_OFFSET_Y))

    def _draw_hold_text(self, screen):
        # draw start hold text
        self._draw_hold_text_helper(self.start_rows[0], self.start_cols[0], 0, screen)
        if self.is_second_start:
            self._draw_hold_text_helper(self.start_rows[1], self.start_cols[1], 0, screen)

        # draw hold text
        for hold_row, hold_col in self.text_holds:
            self._draw_hold_text_helper(hold_row, hold_col, 1, screen)

        # draw end hold text
        self._draw_hold_text_helper(self.end_rows[0], self.end_cols[0], 2, screen)
        if self.is_second_end:
            self._draw_hold_text_helper(self.end_rows[1], self.end_cols[1], 2, screen)
    
    def clear_holds(self):
        self.hold_surf.fill(pygame.Color(0, 0, 0, 0))

    def update_threshold(self, threshold):
        if threshold != self.threshold:
            self.threshold = threshold
            self._redraw_holds()

    def _redraw_holds(self):
        self.clear_holds()
        self.draw_holds()

    def draw(self, screen):
        screen.blit(self.surf, self.rect)
        screen.blit(self.hold_surf, self.rect)
        self._draw_hold_text(screen)


# Variable to keep the main loop running
running = True

# Generate board from model
model_obj = MoonBoardCVAE()
model_obj.load_state_dict(torch.load(MODEL_PATH))
model_obj.eval()

board = Board(model_obj)
board.draw_holds()

slider = Slider(screen, 80, SCREEN_HEIGHT - 140, 380, 20, min=0.05, max=1, step=0.01, initial=0.2)
slider_text = TextBox(screen, 480, SCREEN_HEIGHT - 140, 40, 20, fontSize=12)

grade_slider = Slider(screen, 80, SCREEN_HEIGHT - 100, 380, 20, min=0, max=N_GRADES-1, step=1, initial=2)
grade_slider_text = TextBox(screen, 480, SCREEN_HEIGHT - 100, 40, 20, fontSize=12)

opt_font = pygame.font.SysFont(None, size=24)
hold_text = opt_font.render("Hold", True, pygame.Color(0, 0, 0))
grade_text = opt_font.render("Grade", True, pygame.Color(0, 0, 0))

def button_press():
    board.generate_new_board()
    slider.setValue(0.2)

button = Button(screen, 80, SCREEN_HEIGHT - 60, 380, 40, text="Generate", fontSize=24, pressedColour=(239,223,83))
button.setOnRelease(button_press)

async def main():
    global running, board, slider, slider_text, grade_slider, grade_slider_text, hold_text, grade_text, button
    # global running, board, hold_text, grade_text
    # Main loop
    while running:
        events = pygame.event.get()
        # Look at every event in the queue
        for event in events:
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop.
                if event.key == K_ESCAPE:
                    running = False

            # Did the user click the window close button? If so, stop the loop.
            elif event.type == QUIT:
                running = False
        
        # Fill screen with white
        screen.fill((255, 255, 255))

        # Update the board threshold if slider moved
        slider_text.setText(f"{slider.getValue():.2f}")
        board.update_threshold(slider.getValue())

        grade = GRADE_MAP_REV[grade_slider.getValue()]
        grade_slider_text.setText(grade)
        board.set_grade(grade)

        # Draw the board on the screen
        board.draw(screen)
        screen.blit(hold_text, (16, SCREEN_HEIGHT - 137))
        screen.blit(grade_text, (16, SCREEN_HEIGHT - 97))

        pygame_widgets.update(events)

        # Update the display
        pygame.display.flip()

        await asyncio.sleep(0)  # Let other tasks run

asyncio.run(main())