import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch

from titanic_transformer_mixin import AgeImputer


class TitanicDataset(Dataset):
  def __init__(self,csvpath, mode = 'train'):
        self.mode = mode
        df = pd.read_csv(csvpath)       
        # Data Set Clean up
        # Drop columns with too many missig values or which do not bring additional information
         

        # set missing "Embarked" values to 'S'
        indexes = df.loc[df['Embarked'].isna()].values[:,0]
        for i in indexes:
            df.iloc[i - 1, df.columns.get_loc('Embarked')] = "S"
        print("Adding empty values for 'Embarked'")

        df = df.drop(columns = ['PassengerId', 'Ticket', 'Cabin'])
        print("Dropping columns {}".format(['PassengerId', 'Ticket', 'Cabin']))

        # fill in empty 'Age' entries from AgeImputer
        trans = AgeImputer()
        df = trans.transform(df)

        # mapping 'Names' in data set to numeric values  (<- string)
        self.name_le =  preprocessing.LabelEncoder()
        self.name_le.fit(df['Name'].values)
        df.iloc[:, df.columns.get_loc('Name')] = self.name_le.transform(df['Name'].values)
        print("Mapping 'Name's to values...")

        # mapping 'Sex' in data set to numeric values (<- ("male", "female"))
        self.sex_le =  preprocessing.LabelEncoder()
        self.sex_le.fit(df['Sex'].values)
        df.iloc[:, df.columns.get_loc('Sex')] = self.sex_le.transform(df['Sex'].values)
        print("Mapping 'Sex' column to values..")

        # mapping 'Embarked' in data set to numeric values (<- string)
        self.embarked_le = preprocessing.LabelEncoder()
        self.embarked_le.fit(df['Embarked'].values)
        df.iloc[:, df.columns.get_loc('Embarked')] = self.embarked_le.transform(df['Embarked'].values)
        print("Mapping 'Embarked' column to values..")

        # normalise values using Min-Max 
        #min_max_scaler = preprocessing.MinMaxScaler()
        #x_scaled = min_max_scaler.fit_transform(df.values)
        #df = pd.DataFrame(x_scaled)
      

        #if self.mode == 'train':
            
        #    df = df.dropna()
        #    self.inp = df.iloc[:,1:].values
        #    #self.oup = df.iloc[:, 0].values.reshape(891,1)
        #    self.oup = df.iloc[:,0].values.reshape(np.size(df.iloc[:,1:].values, 0),1)
        #else:
        #    self.inp = df.values
        df = df.dropna()
        self.inp = df.iloc[:,1:].values
        self.oup = df.iloc[:,0].values.reshape(np.size(df.iloc[:,1:].values, 0),1)
        
            
  def __len__(self):
        return len(self.inp)

  def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.from_numpy(self.inp[idx])
            oupt  = torch.from_numpy(self.oup[idx])
            return { 'inp': inpt,
                     'oup': oupt
            }
        else:
            inpt = torch.from_numpy(self.inp[idx])
            oupt  = torch.from_numpy(self.oup[idx])
            return { 'inp': inpt,
                     'oup': oupt
            }
