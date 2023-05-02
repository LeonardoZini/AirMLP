import torch


def create_dataset(dataset, lookback):
        batch = len(dataset) - lookback
        
        X=torch.zeros((batch,lookback,dataset.shape[1]-1))
        y=torch.zeros((batch,lookback,1))
        for i in range(len(dataset)-lookback):
            feature = dataset[i:i+lookback,:-1]
            target = dataset[i+lookback,-1]
            X[i,:,:]=feature
            y[i,:,:]=target.unsqueeze(0).t()
            
        
        
        return X,y


def creation(dataset_, lookback=10, p=0.8):
    '''
    Staring from a pandas.DataFrame object, divide it in train and test set, putting together a number of "lookback" consecutive record.
    Returns a tuple of tuple, where each of the two tuple are the torch.tensor relative to train and test set.
    
    
    '''
    dataset=dataset_.copy()
    train_size = int(len(dataset) * p)
    test_size = len(dataset) - train_size
    if p==1:
         train_size=len(dataset)

    # in futuro bisogna controllare l'eccezione sul drop e non sulla trasformazione in numpy
    try:
        train, test = torch.tensor(dataset.to_numpy()[:train_size]),torch.tensor(dataset.to_numpy()[train_size:])
    except TypeError:
        # Datetime never dropped, drop it now.
        dataset.drop(columns="valid_at",inplace=True)
        train, test = torch.tensor(dataset.to_numpy()[:train_size]),torch.tensor(dataset.to_numpy()[train_size:])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_train, y_train = X_train.to(device), y_train.to(device)
    if p!=1:   
        X_test, y_test = create_dataset(test, lookback=lookback)
        X_test, y_test = X_test.to(device), y_test.to(device)
        return (X_train, y_train),(X_test, y_test)
    else:
        return (X_train,y_train)



     