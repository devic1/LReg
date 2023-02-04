import torch

x = torch.arange(10,dtype=torch.float)
w = torch.tensor(6)
g = torch.tensor(3)
y = x * w + g

k = torch.randn(1,dtype=torch.float,requires_grad=True)
b = torch.randn(1,dtype=torch.float,requires_grad=True)

for i in range(1000):
    out = (x * k) + b
    loss = (abs(out-y)).mean()
    k.grad = None
    b.grad = None
    loss.backward()
    k.data += -0.01*k.grad
    b.data += -0.01*b.grad

print("Linear regression using only tensors")
print(f'The loss at the end {loss:.4f}')
print(f'Expected Weight = {w} and Bias = {g}')
print(f'Derived Weight = {k.item():.4f} and Bias = {b.item():.4f}')
print("Predicting for a integer 100")
print(f'Function : M*w + b ---> M = 100 ; {(torch.tensor(100)*k+b).item()}')

