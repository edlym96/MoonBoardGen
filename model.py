import torch
import numpy as np
from constants import MOONBOARD_COLS, MOONBOARD_ROWS, N_GRADES

MODEL_PATH = "./new_moonboard2016_cvae_64_data_v2.model"


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

    def sample(self, num_samples, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.latent_size).to(device)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples

class MoonBoardCVAE(torch.nn.Module):
    INPUT_SIZE = 3 * MOONBOARD_ROWS * MOONBOARD_COLS
    def __init__(self, num_classes=N_GRADES, latent_size=64):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.INPUT_SIZE + num_classes, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, latent_size),
            torch.nn.ReLU(),
        )
        # Add mu and log_var layers for reparameterization
        self.mu = torch.nn.Linear(latent_size, latent_size)
        self.log_var = torch.nn.Linear(latent_size, latent_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size + num_classes, 256),
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

    def encode(self, x, x_grade_one_hot):
        x_cat = torch.cat((x, x_grade_one_hot),1)
        # Pass the input through the encoder
        encoded = self.encoder(x_cat)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        return encoded, mu, log_var

    def decode(self, z, grade_one_hot):
        z_cat = torch.cat((z, grade_one_hot), 1)
        return self.decoder(z_cat)

    def forward(self, x, x_grade_one_hot):
        encoded, mu, log_var = self.encode(x, x_grade_one_hot)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        z_cat = torch.cat((z, x_grade_one_hot),1)
        # Pass the latent variable through the decoder
        decoded = self.decoder(z_cat)
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples, grades, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            if np.isscalar(grades):
                grades = torch.ones(num_samples, dtype=int) * grades
            grade_one_hot = torch.nn.functional.one_hot(grades, num_classes=N_GRADES)
            
            # Generate random noise
            z = torch.randn(num_samples, self.latent_size).to(device)
            # Pass the noise through the decoder to generate samples
            samples = self.decode(z, grade_one_hot)
        # Return the generated samples
        return samples

def hold_loss(recon_x, min_thresh, max_thresh, weights, hold_thresh=0.5, at_least_weight=1):
    # Penalise reconstructed boards which have more than max_thresh amount
    if max_thresh is not None:
        AT_MOST_ERR = torch.sum((torch.sum((recon_x - hold_thresh).clamp(min=0, max=1e-4) * 1e4, axis=1) - max_thresh).clamp(min=0) * weights)
    else:
        AT_MOST_ERR = 0
    # Penalise reconstructed boards which have less than min_thresh amount
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
    # MOVE_HOLD_ERR = hold_loss(recon_x * move_hold_mask, 3, None, flattened_weights)
    
    end_hold_mask = torch.zeros(recon_x.size(1))
    end_hold_mask[2*(MOONBOARD_ROWS*MOONBOARD_COLS):] = 1
    END_HOLD_ERR = hold_loss(recon_x * end_hold_mask, 1, 2, flattened_weights)
    
    # Combine the losses by adding them together and return the result
    return BCE + 0.1 * KLD + START_HOLD_ERR + END_HOLD_ERR
