import torch
import pickle
import datetime

from dataset import MoonBoardWeightedDataset
from parse_problems import MOONBOARD_COLS, MOONBOARD_ROWS

PATH = "./new_moonboard2016_vae_64_hold_err_v2.model"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
TODO:
 - Change sampling methodology, clean the data better
 - Add conditional VAE with grades
 - Rewrite data loader to allow for grades
 - clean up code
"""

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Set the number of hidden units
        self.num_hidden = 64
        
        # Define the encoder part of the autoencoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(594, 256),  # input size: 594, output size: 256
            torch.nn.ReLU(),  # apply the ReLU activation function
            torch.nn.Linear(256, self.num_hidden),  # input size: 256, output size: num_hidden
            torch.nn.ReLU(),  # apply the ReLU activation function
        )
        
        # Define the decoder part of the autoencoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden, 256),  # input size: num_hidden, output size: 256
            torch.nn.ReLU(),  # apply the ReLU activation function
            torch.nn.Linear(256, 594),  # input size: 256, output size: 594
            torch.nn.Sigmoid(),  # apply the sigmoid activation function to compress the output to a range of (0, 1)
        )

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        # Return both the encoded representation and the reconstructed output
        return encoded, decoded
    

class VAE(AutoEncoder):
    def __init__(self):
        super().__init__()
        # Add mu and log_var layers for reparameterization
        self.mu = torch.nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = torch.nn.Linear(self.num_hidden, self.num_hidden)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.num_hidden).to(device)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples

class MoonBoardVAE(torch.nn.Module):
    INPUT_SIZE = 3 * MOONBOARD_ROWS * MOONBOARD_COLS
    def __init__(self, latent_size=64):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.INPUT_SIZE, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, latent_size),
            torch.nn.ReLU(),
        )
        # Add mu and log_var layers for reparameterization
        self.mu = torch.nn.Linear(latent_size, latent_size)
        self.log_var = torch.nn.Linear(latent_size, latent_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.INPUT_SIZE),
            torch.nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.latent_size).to(device)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples

def hold_loss(recon_x, min_thresh, max_thresh, weights, hold_thresh=0.5, at_least_weight=1):

    # Penalise reconstructed boards which have more than max_thresh amount
    # AT_MOST_ERR = torch.sum((torch.sum(recon_x, axis=1) - max_thresh).clamp(min=0) * weights)
    AT_MOST_ERR = torch.sum((torch.sum((recon_x - hold_thresh).clamp(min=0, max=1e-4) * 1e4, axis=1) - max_thresh).clamp(min=0) * weights)
    # Penalise reconstructed boards which have less than min_thresh amount
    # AT_LEAST_ERR = torch.sum(((-1 * torch.sum(recon_x, axis=1)) + min_thresh).clamp(min=0) * weights)
    AT_LEAST_ERR = torch.sum((min_thresh - torch.sum((recon_x - hold_thresh).clamp(min=0, max=1e-4) * 1e4, axis=1)).clamp(min=0) * weights)

    return at_least_weight * AT_LEAST_ERR + AT_MOST_ERR

# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def loss_function(recon_x, x, mu, logvar, weights):
    # Compute the binary cross-entropy loss between the reconstructed output and the input data
    bce_loss = torch.nn.BCELoss(reduction='none')
    # bce_loss = torch.nn.BCELoss(reduction='none')
    BCE = torch.sum(bce_loss(recon_x, x) * weights)
    # Compute the Kullback-Leibler divergence between the learned latent variable distribution and a standard Gaussian distribution
    KLD = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()) * weights)
    flattened_weights = weights.flatten()
    start_hold_mask = torch.zeros(recon_x.size(1))
    start_hold_mask[:(MOONBOARD_ROWS*MOONBOARD_COLS)] = 1
    START_HOLD_ERR = hold_loss(recon_x * start_hold_mask, 1, 2, flattened_weights)
    
    # move_hold_mask = torch.zeros(recon_x.size(1))
    # move_hold_mask[(MOONBOARD_ROWS*MOONBOARD_COLS):2*(MOONBOARD_ROWS*MOONBOARD_COLS)] = 1
    # MOVE_HOLD_ERR = hold_loss(recon_x * move_hold_mask, 3, 10, flattened_weights, at_least_weight=3)
    
    end_hold_mask = torch.zeros(recon_x.size(1))
    end_hold_mask[2*(MOONBOARD_ROWS*MOONBOARD_COLS):] = 1
    END_HOLD_ERR = hold_loss(recon_x * end_hold_mask, 1, 2, flattened_weights)
    
    # Combine the two losses by adding them together and return the result
    # return 10 * BCE + KLD + 50 * MOVE_HOLD_ERR + 100 * START_HOLD_ERR + 50 * END_HOLD_ERR
    # possible drop the 2*
    return BCE + 0.05 * KLD + START_HOLD_ERR + END_HOLD_ERR
    # return BCE + 0.05 * KLD + 2 * START_HOLD_ERR + END_HOLD_ERR + MOVE_HOLD_ERR


def train_vae(X_train, X_weights, learning_rate=3e-4, num_epochs=100, batch_size=64, model_path=None):
    # Convert the training data to PyTorch tensors
    X_train = torch.from_numpy(X_train).to(device)
    X_weights = torch.from_numpy(X_weights).to(device)
    
    # Create the autoencoder model and optimizer
    model = MoonBoardVAE()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Add LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    # Set the device to GPU if available, otherwise use CPU
    model.to(device)

    # Create a DataLoader to handle batching of the training data
    train_dataset = MoonBoardWeightedDataset(X_train, X_weights)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    min_loss = float('inf')
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, weights) in enumerate(train_loader):
            # Get a batch of training data and move it to the device
            data = data.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()            
            # Forward pass
            encoded, decoded, mu, log_var = model(data)

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
            torch.save(model.state_dict(), PATH)
            min_loss = epoch_loss
        scheduler.step(epoch_loss)

    # Return the trained model
    return model

if __name__ == "__main__":
    with open('./data/train.pkl', 'rb') as f:
        train, train_is_bench = pickle.load(f)
    train = train.astype('float32')

    with open('./data/test.pkl', 'rb') as f:
        test, test_is_bench = pickle.load(f)

    train_vae(train, train_is_bench)