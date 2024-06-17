import torch

class BilinearInterpolation:
    def __init__(self, x, y, x_values, y_values):
        # Sort x and y values
        x_values, _ = torch.sort(x_values)
        y_values, _ = torch.sort(y_values)
        x, _ = torch.sort(x)
        y, _ = torch.sort(y)

        # Clamp x and y values to the range of x_values and y_values
        x = torch.clamp(x, min=x_values.min(), max=x_values.max())
        y = torch.clamp(y, min=y_values.min(), max=y_values.max())
    
        # Find the indices of the nearest lower values in x_values and y_values
        # for each point in x and y
        x_idx = torch.searchsorted(x_values, x, right=True) - 1
        y_idx = torch.searchsorted(y_values, y, right=True) - 1
        
        # Clip indices to be within the valid range
        x_idx = torch.clamp(x_idx, 0, len(x_values) - 2)
        y_idx = torch.clamp(y_idx, 0, len(y_values) - 2)
        
        # Get the x and y values of the corners surrounding each (x, y) point
        x1 = x_values[x_idx]
        x2 = x_values[x_idx + 1]
        y1 = y_values[y_idx]
        y2 = y_values[y_idx + 1]

        # Compute the weights for bilinear interpolation
        wx1 = (x2 - x) / (x2 - x1)
        wx2 = (x - x1) / (x2 - x1)
        wy1 = (y2 - y) / (y2 - y1)
        wy2 = (y - y1) / (y2 - y1)

        # Reshape weights for broadcasting
        self.wx1 = wx1.view(1, 1, -1, 1)
        self.wx2 = wx2.view(1, 1, -1, 1)
        self.wy1 = wy1.view(1, 1, 1, -1)
        self.wy2 = wy2.view(1, 1, 1, -1)

        # Expand dimensions for broadcasting
        self.x_idx = x_idx.unsqueeze(-1)
        self.y_idx = y_idx.unsqueeze(0)


    def __call__(self, z_values):
        # Get the z values of the corners surrounding each (x, y) point
        Q11 = z_values[..., self.x_idx, self.y_idx]
        Q12 = z_values[..., self.x_idx, self.y_idx + 1]
        Q21 = z_values[..., self.x_idx + 1, self.y_idx]
        Q22 = z_values[..., self.x_idx + 1, self.y_idx + 1]
        
        # Linear interpolation in y
        R1 = self.wx1 * Q11 + self.wx2 * Q21
        R2 = self.wx1 * Q12 + self.wx2 * Q22
        
        # Linear interpolation in x
        z = self.wy1 * R1 + self.wy2 * R2

        return z