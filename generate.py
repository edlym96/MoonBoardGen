import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from model import MODEL_PATH, MoonBoardCVAE
from constants import GRADE_MAP
from plotting import MOONBOARD_IMG, plot_board_helper
import torch

def generate_board(model, grade):
    # translate grade to index
    grade_idx = GRADE_MAP[grade]
    # Sample a board from the model, reshape and convert to numpy array
    generated_board = model.sample(1, grade_idx).reshape(3, 18, 11).numpy()
    return generated_board

def generate_and_plot(model):
    # Global state grade for updating plots, init to 6B+
    global_grade = '6B+'
    img = plt.imread(MOONBOARD_IMG)

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1, figsize=(10,9))
    ax.set_aspect('equal')
    axwave = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    # "global" variable to hold slider reference
    slider = None
    
    def new_board(event):
        nonlocal slider
        print(f"Generating board with grade {global_grade}")
        generated_board = generate_board(model, global_grade)
        # Clear axis states
        ax.cla()
        axwave.cla()
        # Keep reference to slider object
        slider = plot_board_helper(generated_board, fig, ax, axwave, img)

    # Init board plot
    new_board(global_grade)

    # Set grade callback function for radio buttons
    def set_grade(grade):
        nonlocal global_grade
        global_grade = grade

    # radio buttons
    rax = plt.axes([0.12, 0.15, 0.1, 0.2])
    radio_button = RadioButtons(rax, list(GRADE_MAP.keys()), active=GRADE_MAP[global_grade], activecolor='black')
    radio_button.on_clicked(set_grade)

    # Generate button
    button_axes = fig.add_axes([0.66, 0.000001, 0.08, 0.04])
    bnext = Button(button_axes, 'Generate',color="yellow")
    bnext.on_clicked(new_board)
    plt.show()

if __name__ == "__main__":
    model_obj = MoonBoardCVAE()
    model_obj.load_state_dict(torch.load(MODEL_PATH))
    model_obj.eval()
    generate_and_plot(model_obj)
