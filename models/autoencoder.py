
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Convolutional Variational Autoencoder.
    Compresses (B, 1, 32, 32) -> Latent (B, 4, 4, 4).
    """
    def __init__(self, input_dim=1, latent_dim=4, hidden_dims=[32, 64, 128]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        modules = []
        in_channels = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Latent Space (Mean and LogVar)
        # 32 -> 16 -> 8 -> 4 (after 3 strides of 2)
        # Final spatial dim is 4x4
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
        
        # Decoder
        modules = []
        self.decoder_input = nn.ConvTranspose2d(latent_dim, hidden_dims[-1], kernel_size=3, padding=1)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=3, padding=1),
            nn.Softplus() # Ensure positive IV
        )

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return recons, x, mu, log_var

    def loss_function(self, recons, input, mu, log_var, kld_weight=0.00025):
        """
        Computes VAE loss = MSE + kld_weight * KLD
        """
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
        
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
