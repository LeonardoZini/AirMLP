import torch.nn as nn
import torch 


class AirModel(nn.Module):
    '''
    Model based on LSTM.
    '''
    def __init__(self, num_hidden=50, num_fin=36, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_fin, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(num_hidden, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        
        return torch.flatten(x,0)

class AirMLP(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP, self).__init__()
    
    
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin),
                    nn.Dropout(0.7),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),  
                                                  
                    nn.Linear(num_hidden, 1)
    )
  
  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0)
    