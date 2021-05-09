import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class AgeImputer(TransformerMixin):
    def __init__(self):
        """
        Imputes ages of passengers in the Titanic, values to be imputed will be dependant 
        on passenger titles and the presence of parents or children on board
        """
        pass

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        def value_imp(passengers):
            """
            Imputes an age, based on a weighted random choice derived from the non
            null entries in the subsets of the dataset.
            """
            passengers=passengers.copy()
            # Create 3 year age bins
            bins = np.arange(0,passengers['Age'].max()+3,step=3)
            # Assign each passenger an age bin
            passengers['age_bins'] = pd.cut(passengers['Age'],bins=bins,labels=bins[:-1]+1.5)
            # Count totals of age bins
            count = passengers.groupby('age_bins')['age_bins'].count()
            # Assign each age bin a weight
            weights = count/len(passengers['Age'].dropna())
            null = passengers['Age'].isna()
            # For each missing value, give the passenger an age from the age bins available
            passengers.loc[passengers['Age'].isna(),'Age']=np.random.RandomState(seed=42).choice(weights.index,
                           p=weights.values,size=len(passengers[null]))
            return passengers
        master = X.loc[X['Name'].str.contains('Master')]
        mrs = X.loc[X['Name'].str.contains('Mrs')]
        miss = X.loc[X['Name'].str.contains('Miss')]
        no_parch = X.loc[X['Parch']==0]
        parch = X.loc[X['Parch']!=0]
        miss_no_parch = miss.drop([x for x in miss.index if x in parch.index])
        miss_parch = miss.drop([x for x in miss.index if x in no_parch.index])
        remaining_mr = X.loc[X['Name'].str.contains('Mr. ')]
        # Imputing 'Mrs' first, as in cases where passengers have the titles
        # 'Miss' and 'Mrs', they are married so will be in the older category
        name_cats = [master,mrs,miss_no_parch,miss_parch,remaining_mr]
        for name in name_cats:
            X.loc[name.index] = value_imp(name)
        return X
