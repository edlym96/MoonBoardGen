from model import VAE, PATH, MoonBoardVAE
from plotting import plot_board_2016
import torch


model = MoonBoardVAE()
model.load_state_dict(torch.load(PATH))
model.eval()
generated_board = model.sample(1).reshape(3, 18, 11).numpy()
import pdb;pdb.set_trace()