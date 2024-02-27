import pickle
import torch
import datetime

from constants import TEST_PATH, TRAIN_PATH
from dataset import MoonBoardWeightedDataset
from model import MODEL_PATH, MoonBoardCVAE, loss_function

LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
BATCH_SIZE = 64

def train_vae(X_train, X_weights, X_grades, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, model_path=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert the training data to PyTorch tensors
    X_train = torch.from_numpy(X_train).to(device)
    X_weights = torch.from_numpy(X_weights).to(device)
    X_grades = torch.from_numpy(X_grades).to(device)

    # Create the autoencoder model and optimizer
    model_obj = MoonBoardCVAE()
    if model_path:
        model_obj.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=learning_rate)

    # Add LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    # Set the device to GPU if available, otherwise use CPU
    model_obj.to(device)

    # Create a DataLoader to handle batching of the training data
    train_dataset = MoonBoardWeightedDataset(X_train, X_weights, X_grades)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    model_obj.train()
    min_loss = float('inf')
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, weights, grade_one_hot) in enumerate(train_loader):
            # Get a batch of training data and move it to the device
            data = data.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            # Forward pass
            encoded, decoded, mu, log_var = model_obj(data, grade_one_hot)

            # Compute the loss and perform backpropagation
            loss = loss_function(decoded, data, mu, log_var, weights)
            loss.backward()
            optimizer.step()

            # Update the running loss
            total_loss += loss.item()

        # Print the epoch loss
        epoch_loss = total_loss / len(train_loader.dataset)
        print(
            "[{}] Epoch {}/{}: loss={:.4f}".format(datetime.datetime.now(), epoch + 1, num_epochs, epoch_loss)
        )

        if epoch_loss < min_loss:
            print(f"Saving model at epoch {epoch+1}")
            torch.save(model_obj.state_dict(), MODEL_PATH)
            min_loss = epoch_loss
        scheduler.step(epoch_loss)

    # Return the trained model
    return model_obj

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(TRAIN_PATH, 'rb') as f:
        train, train_is_bench, train_grades = pickle.load(f)
    train = train.astype('float32')

    with open(TEST_PATH, 'rb') as f:
        test, test_is_bench, test_grades = pickle.load(f)

    train_vae(train, train_is_bench, train_grades, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device)