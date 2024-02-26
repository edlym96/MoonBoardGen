from dataset import MoonBoardWeightedDataset
from model import PATH, MoonBoardVAE, MoonBoardCVAE, loss_function
from parse_problems import TEST_PATH, TRAIN_PATH
from plotting import plot_board_2016
import torch
import pickle


model_paths = [
    './new_moonboard2016_cvae_64.model',
    './new_moonboard2016_cvae_64_no_hold_err.model',
    './new_moonboard2016_cvae_64_move_hold_err.model',
    './new_moonboard2016_cvae_64_hold_zero_err.model',
    './new_moonboard2016_cvae_64_kll_anneal.model',
    "./new_moonboard2016_cvae_64_data_v2.model"
]

with open(TRAIN_PATH, 'rb') as f:
    train, train_is_bench, train_grades = pickle.load(f)
    train = train.astype('float32')

train = train.astype('float32')
train = torch.from_numpy(train)
train_weights = torch.from_numpy(train_is_bench)
train_grades = torch.from_numpy(train_grades)
train_dataset = MoonBoardWeightedDataset(train, train_weights, train_grades)

with open(TEST_PATH, 'rb') as f:
    test, test_is_bench, test_grades = pickle.load(f)

test = test.astype('float32')
test = torch.from_numpy(test)
test_weights = torch.from_numpy(test_is_bench)
test_grades = torch.from_numpy(test_grades)
test_dataset = MoonBoardWeightedDataset(test, test_weights, test_grades)

# model = MoonBoardVAE()
for path in model_paths:
    model = MoonBoardCVAE()
    model.load_state_dict(torch.load(path))
    model.eval()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    for (data, weights, grade_one_hot) in train_loader:
        encoded, decoded, mu, log_var = model(data, grade_one_hot)

        # Compute the loss and perform backpropagation
        loss = loss_function(decoded, data, mu, log_var, weights)
        train_loss = loss.item()
    train_loss /= len(train_loader.dataset)

    for (data, weights, grade_one_hot) in test_loader:
        encoded, decoded, mu, log_var = model(data, grade_one_hot)

        # Compute the loss and perform backpropagation
        loss = loss_function(decoded, data, mu, log_var, weights)
        test_loss = loss.item()
    test_loss /= len(test_loader.dataset)

    print(f"[{path}]\nTrain loss: {train_loss}\nTest loss {test_loss}\n")

generated_board = model.sample(1, 2).reshape(3, 18, 11).numpy()
import pdb;pdb.set_trace()