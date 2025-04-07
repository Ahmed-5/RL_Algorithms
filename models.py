from torch import nn

class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.unsqueeze = False

        self.layers = []
        if len(input_size) == 3 or len(input_size) == 2:
            self._build_conv_network(input_size)
        elif len(input_size) == 1:
            self._build_fc_network(input_size)

        self.layers = nn.Sequential(*self.layers)

    def _build_conv_network(self, input_size):
        self.unsqueeze = len(input_size) == 2
        channels = 1 if len(input_size) == 2 else input_size[0]
        h = input_size[-2]
        w = input_size[-1]
        
        # First conv block
        self.layers.append(nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())
        x, y = (h + 1) // 2, (w + 1) // 2
        
        # Second conv block
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())
        x, y = (x + 1) // 2, (y + 1) // 2
        
        # Third conv block
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.ReLU())
        x, y = (x + 1) // 2, (y + 1) // 2
        
        # FC layers
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(x * y * 128, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(512, self.n_actions))

    def _build_fc_network(self, input_size):
        self.layers.append(nn.Linear(input_size[0], 128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, self.n_actions))

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(1)
        return self.layers(x)