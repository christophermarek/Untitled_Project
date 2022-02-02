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

    name = 'BlackScholesModel_Simple2Layer'

    def __init__(self, numHiddenNeurons):
        input_size = 5
        hidden_sizes = [25, 25]
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

class BlackScholesModel_Simple_Greeks_1layer(nn.Module):
    
    name = 'BlackScholesModel_Simple_Greeks_1layer'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 5
        super(BlackScholesModel_Simple_Greeks_1layer, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x

class BlackScholesModel_Simple_Greeks_2layer(nn.Module):
    
    name = 'BlackScholesModel_Simple_Greeks_2layer'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 5
        super(BlackScholesModel_Simple_Greeks_2layer, self).__init__()
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

class BlackScholesModel_Simple_Greeks_4layer(nn.Module):
    
    name = 'BlackScholesModel_Simple_Greeks_4layer'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 5
        super(BlackScholesModel_Simple_Greeks_4layer, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(10, 50),
                                   nn.ReLU(),
                                   nn.Linear(50, 25),
                                   nn.ReLU(),
                                   nn.Linear(25, 10),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x


class BS_DELTA_SIMPLE(nn.Module):
    
    name = 'BS_DELTA_SIMPLE'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 1
        super(BS_DELTA_SIMPLE, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x
    
class BS_GAMMA_SIMPLE(nn.Module):
    
    name = 'BS_GAMMA_SIMPLE'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 1
        super(BS_GAMMA_SIMPLE, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x
    
class BS_RHO_SIMPLE(nn.Module):
    
    name = 'BS_RHO_SIMPLE'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [10, 10]
        output_size = 1
        super(BS_RHO_SIMPLE, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size),
                                   )
    def forward(self, input):
        x = self.model(input)
        return x
    
class BS_DELTA_SIMPLE2(nn.Module):
    
    name = 'BS_DELTA_SIMPLE2'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [50, 50]
        output_size = 1
        super(BS_DELTA_SIMPLE2, self).__init__()
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
    
class BS_GAMMA_SIMPLE2(nn.Module):
    
    name = 'BS_GAMMA_SIMPLE2'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [50, 50]
        output_size = 1
        super(BS_GAMMA_SIMPLE2, self).__init__()
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
    
class BS_RHO_SIMPLE2(nn.Module):
    
    name = 'BS_RHO_SIMPLE2'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [50, 50]
        output_size = 1
        super(BS_RHO_SIMPLE2, self).__init__()
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

class BS_THETA_SIMPLE2(nn.Module):
    
    name = 'BS_THETA_SIMPLE'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [50, 50]
        output_size = 1
        super(BS_THETA_SIMPLE2, self).__init__()
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
    
class BS_VEGA_SIMPLE2(nn.Module):
    
    name = 'BS_VEGA_SIMPLE'
    
    def __init__(self):
        input_size = 5
        hidden_sizes = [50, 50]
        output_size = 1
        super(BS_VEGA_SIMPLE2, self).__init__()
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