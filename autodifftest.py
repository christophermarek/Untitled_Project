import torch
import matplotlib.pyplot as plt

x = torch.linspace(-1, 1, 100).unsqueeze(1)
y_prime = 3 * x**2 + torch.normal(torch.zeros(100,1), torch.ones(100,1) * 0.1)
y = y_prime / 3 * x

model = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.Sigmoid(),
    torch.nn.Linear(32, 1)
)

opt = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(1000):
    opt.zero_grad()
    loss = torch.mean((model(x) - y)**2)
    loss.backward()
    opt.step()

with torch.no_grad():
    fig, (ax1, ax2) = plt.subplots(2, 1, num=1)
    ax1.plot(x.numpy(), y.numpy(), '.', label='data')
    ax1.plot(x.numpy(), model(x).numpy(), label='model')
    ax2.plot(x.numpy(), y_prime.numpy(), '.', label='data deriv')

x.requires_grad = True
z = model(x)
z.sum().backward()

with torch.no_grad():
    ax2.plot(x.numpy(), x.grad.numpy(), label='model deriv')

ax1.legend()
ax2.legend()
plt.show()