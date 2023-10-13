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


class AirMLP_5(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP_5, self).__init__()
    
    # nn.Dropuout(0.5) -> usare anche sigmoide
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin,affine=False),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(), 
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),                   
                    nn.Linear(num_hidden, 1)
    )
  
  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0)
    
class AirMLP_6(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP_6, self).__init__()
    
    # nn.Dropuout(0.5) -> usare anche sigmoide
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin,affine=True),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),                             
                    nn.Linear(num_hidden, 1)
    )
  
  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0)
  
class AirMLP_7(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP_7, self).__init__()
    
    # nn.Dropuout(0.5) -> usare anche sigmoide
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin,affine=True),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),  
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),                           
                    nn.Linear(num_hidden, 1)
    )
  
  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0).unsqueeze(1)
    
  
class AirMLP_8(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP_8, self).__init__()
    
    # nn.Dropuout(0.5) -> usare anche sigmoide
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin,affine=True),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),  
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),                             
                    nn.Linear(num_hidden, 1)
    )

  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0)
  
class AirMLP_7h(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP_7h, self).__init__()
    
    # nn.Dropuout(0.5) -> usare anche sigmoide
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin,affine=True),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),  
                    nn.Linear(num_hidden, num_hidden//2),  
                    nn.ReLU(),                           
                    nn.Linear(num_hidden//2, 1)
    )
  
  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0)

class AirMLP_8h(nn.Module):
  '''
  Model based on Linear Layer

  output of the netowrk is a tensor of shape [BATCH_SIZE]
  If you want to put in the shape [BATCH_SIZE,1] remove the flatten function in the forward and adjust the input ground truth
  '''
  def __init__(self, num_fin: int, num_hidden: int):
    super(AirMLP_8h, self).__init__()
    
    # nn.Dropuout(0.5) -> usare anche sigmoide
    self.net = nn.Sequential(
                    nn.BatchNorm1d(num_fin,affine=True),
                    nn.Linear(num_fin, num_hidden),  
                    nn.ReLU(),      
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),  
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),    
                    nn.Linear(num_hidden, num_hidden),  
                    nn.ReLU(),   
                    nn.Linear(num_hidden, num_hidden//2),  
                    nn.ReLU(),                             
                    nn.Linear(num_hidden//2, 1)
    )

  def forward(self, x: torch.Tensor):
    output = self.net(x)
    return torch.flatten(output,0)