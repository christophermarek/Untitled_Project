from torch import nn, optim

class BlackScholesModel_Simple(nn.Module):
    
    def __init__(self, numHiddenNeurons):
        input_size = 5
        hidden_sizes = [numHiddenNeurons, numHiddenNeurons]
        output_size = 1
        super(BlackScholesModel_Simple, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x

class BlackScholesModel_Simple2Layer(nn.Module):

    def __init__(self, numHiddenNeurons):
        input_size = 5
        hidden_sizes = [numHiddenNeurons, numHiddenNeurons]
        output_size = 1
        super(BlackScholesModel_Simple2Layer, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x


class BlackScholesModel_Simple3Layer(nn.Module):

    def __init__(self, numHiddenNeurons):
        input_size = 5
        hidden_sizes = [numHiddenNeurons, numHiddenNeurons]
        output_size = 1
        super(BlackScholesModel_Simple3Layer, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x


class BlackScholesModel_Simple4Layer(nn.Module):

    def __init__(self, numHiddenNeurons):
        input_size = 5
        hidden_sizes = [numHiddenNeurons, numHiddenNeurons]
        output_size = 1
        super(BlackScholesModel_Simple4Layer, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x
