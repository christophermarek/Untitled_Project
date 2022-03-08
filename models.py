from torch import nn, optim

class BS_VEGA_SIMPLE2(nn.Module):
    
    name = 'BS_VEGA_SIMPLE'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [50, 50]
        output_size = 1
        super(BS_VEGA_SIMPLE2, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    # do i need this? i never call it
    def forward(self, input):
        x = self.model(input)
        return x