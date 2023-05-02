import pandas as pd
ARI1952 = r"./data/ari-1952.csv" # 1
ARI1953 = r"./data/ari-1953.csv" # 1
ARI2049 = r"./data/ari-2049.csv" # 2
ARI1885 = r"./data/ari-1885.csv" # 2
ARI1727 = r"./data/ari-1727.csv" # 1
ARPA = r"./data/arpa.csv"        # GT


class DataCollection:
    def __init__(self, drop_null:bool=True):

        '''
        Initialization of class, must be refactored and make it scalable but for now it is ok
        '''
        self.ari1952 = pd.read_csv(ARI1952).drop(columns=["pm1","pm4","pm10","wind_direction"])
        self.ari1953 = pd.read_csv(ARI1953).drop(columns=["pm1","pm4","pm10","wind_direction"])
        self.ari2049 = pd.read_csv(ARI2049).drop(columns=["pm1","pm4","pm10","wind_direction"])
        self.ari1885 = pd.read_csv(ARI1885).drop(columns=["pm1","pm4","pm10","wind_direction"])
        self.ari1727 = pd.read_csv(ARI1727).drop(columns=["pm1","pm4","pm10","wind_direction"])
        

        self.ari1952["valid_at"] = pd.to_datetime(self.ari1952["valid_at"]).dt.round("H")
        self.ari1953["valid_at"] = pd.to_datetime(self.ari1953["valid_at"]).dt.round("H")
        self.ari1885["valid_at"] = pd.to_datetime(self.ari1885["valid_at"]).dt.round("H")
        self.ari1727["valid_at"] = pd.to_datetime(self.ari1727["valid_at"]).dt.round("H")
        self.ari2049["valid_at"] = pd.to_datetime(self.ari2049["valid_at"]).dt.round("H")
        

        self.wiseair = [self.ari1952,self.ari1953,self.ari1885,self.ari1727,self.ari2049]
        #self.wiseair = [self.ari1952,self.ari1953,self.ari1727]

        self.arpa = pd.read_csv(ARPA)
        self.arpa["valid_at"] = pd.to_datetime(self.arpa["valid_at"]).dt.round("H")
        if drop_null:
            self.arpa.dropna(inplace=True)
            
        self.datetime=True

    def get_devices(self):
        return self.wiseair.copy()

    def get_gt(self):
        return self.arpa.copy()

    

    def get_dataset(self,drop_datetime:bool=False):  
        '''
        This function returns the dataset, composed by the union of all wiseair's device merged with the ground truth data, in
        order to don't lose the reference to the right data, and the ground truth data itself.
        '''
        self.dataset = pd.DataFrame()
        for k,i in enumerate(self.get_devices()):
            
            tmp = pd.merge(i,self.arpa,how="inner",on="valid_at").rename(columns={"pm2p5_y":"pm2p5_t","pm2p5_x":"pm2p5"})
            tmp["dev"]=k
            self.dataset = pd.concat([tmp,self.dataset])
       
        self.dataset.sort_values(by=["dev","valid_at"],inplace=True)
        self.dataset.reset_index(drop=True, inplace=True)
        self.dataset.drop(columns="dev",inplace=True)
        if drop_datetime:
            self.datetime=False
            return (self.dataset.drop(columns="valid_at"),self.arpa.drop(columns="valid_at"))
        
        return (self.dataset,self.arpa) # Return dataset complete device, gt

    
        
