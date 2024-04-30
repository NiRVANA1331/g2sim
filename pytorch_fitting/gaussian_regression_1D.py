import numpy as np
import matplotlib.pyplot as plt
import torch

class Env():
    def __init__(self, mu, sigma):
        self.x = torch.linspace(-10,10,10)
        self.gt = (1/2) * torch.log(torch.tensor([2]) * torch.tensor(torch.pi) * torch.tensor(sigma**2)) + ((self.x-mu)**2)/(2*torch.tensor(sigma**2))
    
    def plot(self):
        plt.scatter(self.x, self.gt)
        plt.show()

class Gaussian_Regression_Model(torch.nn.Module):
    def __init__(self, mu_start, sigma_start):      
        super().__init__()  
        self.mu = torch.nn.Parameter(torch.tensor(mu_start))
        self.sigma = torch.nn.Parameter(torch.tensor(sigma_start))

    def forward(self, x):
        return (1/2) * torch.log(2*torch.pi*self.sigma**2) + ((x-self.mu)**2)/(2*self.sigma**2)
    
    def string(self):
        return f'Mu = {self.mu.item()}, Sigma = {self.sigma.item()}'
    
def train(env, epochs, mu_start, sigma_start):
    model = Gaussian_Regression_Model(mu_start, sigma_start)
    loss_funct = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for param in model.parameters():
        print(param)

    loss_array = []
    for t in range(epochs):
        y_pred = model(env.x)
        loss = loss_funct(y_pred, env.gt)
        loss_array.append(loss.item())
        if t % 100 == 99:
            print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Result: {model.string()}')
    plt.plot(loss_array)
    plt.show()
