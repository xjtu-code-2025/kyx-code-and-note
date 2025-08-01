from abc import abstractmethod
from matplotlib import pyplot as plt, transforms
import torch
from torchvision import datasets, transforms
from torch import  nn
from torch.nn import functional as F
from typing import List, Callable, Optional, Union, Any, TypeVar, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
from torchvision.datasets import MNIST
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')
class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class CVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 num_classes: int = 10,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4+num_classes, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4+num_classes, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim+num_classes, hidden_dims[-1] * 4 )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor,labels:Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        #加入类别信息作为条件
        label_onehot = F.one_hot(labels, num_classes=self.num_classes).float().to(result.device)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        result = torch.cat((result, label_onehot), dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor,labels:Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #同理解码器也拼接onehot编码
        label_onehot = F.one_hot(labels, num_classes=self.num_classes).float().to(z.device)
        z = torch.cat((z, label_onehot), dim=1)
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor,labels:Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input,labels)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z,labels), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor,labels:Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x,labels)[0]

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 若模型设计为输入128x128，可改成64x64适配
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 将单通道扩展为3通道以适配输入为C=3的模型
])

mnist_train = MNIST(
    root='./data_mnist',
    train=True,
    download=True,
    transform=transform
)

mnist_test = MNIST(
    root='./data_mnist',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

model = CVAE(in_channels=3, latent_dim=128)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(2):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        recons, input, mu, log_var = model(data,labels)
        loss_dict = model.loss_function(recons, input, mu, log_var, M_N=1.0)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, labels = batch 
        labels = labels.to(device) 
        x = x.to(device)
        recons = model.generate(x,labels) 
        break  
n = 8
x = x[:n].cpu()
recons = recons[:n].cpu()

fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

for i in range(n):
    axes[0, i].imshow(x[i].permute(1, 2, 0))       # 原图
    axes[0, i].axis('off')
    axes[1, i].imshow(recons[i].permute(1, 2, 0))   # 重建图
    axes[1, i].axis('off')

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")
plt.tight_layout()
plt.show()