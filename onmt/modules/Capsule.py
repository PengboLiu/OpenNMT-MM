import torch 
import torch.nn as nn

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)

        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).cuda()

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)

        return v


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        # self.conv = nn.Conv2d(1, 256, 9)
        self.relu = nn.ReLU(inplace=True)

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=512,
                                        out_channels=8,
                                        kernel_size=3,
                                        stride=2)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=288,
                                    num_caps=10,
                                    dim_caps=16,
                                    num_routing=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, 512, 7, 7)

        out = self.relu(x)
        out = self.primary_caps(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, num_capsules)

        return out