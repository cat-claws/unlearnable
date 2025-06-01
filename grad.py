import torch

# Define the function f(x, y)
def f(x, y):
    return x * y ** 0.3 # Example: replace with any differentiable function

# Constants
y_val = 3.0
x1_val = 2.0

# Compute df/dy at x1 (constant)
x1 = torch.tensor(x1_val, requires_grad=True)
y = torch.tensor(y_val, requires_grad=True)
f1 = f(x1, y)
df_dy_x1 = torch.autograd.grad(f1, y)[0].item()  # Detach as scalar

# Optimize x2 to satisfy df/dy(x2, y) ≈ -df_dy(x1, y)
x2 = torch.tensor(-1.0, requires_grad=True)
optimizer = torch.optim.SGD([x2], lr=0.1)

for _ in range(100):
    optimizer.zero_grad()

    # Recreate y for each iteration with grad
    y = torch.tensor(y_val, requires_grad=True)

    f2 = f(x2, y)
    df_dy_x2 = torch.autograd.grad(f2, y, create_graph=True)[0]

    # Loss: match negative gradient
    loss = (df_dy_x2 + df_dy_x1) ** 2

    print(df_dy_x2)

    loss.backward()
    optimizer.step()

# Final results
print(f"x2 found: {x2.item():.4f}")
print(f"df/dy at x1: {df_dy_x1:.4f}")
y = torch.tensor(y_val, requires_grad=True)
print(f"df/dy at x2: {torch.autograd.grad(f(x2, y), y)[0].item():.4f}")
