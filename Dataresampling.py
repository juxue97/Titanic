from collections import Counter

#undersampling
from imblearn.under_sampling import TomekLinks,ClusterCentroids

class Undersampler:
    #initialize properties
    X=None
    y=None

    def __init__(self,X,y):
        self.X=X
        self.y=y

        print('US Method: [TL,CC]')
    
    #display
    def print_result(self,y_after):
        print(f'After: {Counter(y_after)} \n')

    #TomekLinks
    def TL(self):
        tomek =TomekLinks(sampling_strategy='majority',n_jobs=-1)
        X_res_down,y_res_down = tomek.fit_resample(self.X,self.y)

        print('TomekLinks')
        self.print_result(y_res_down)

        return X_res_down,y_res_down
    
    def CC(self):
        cc = ClusterCentroids(sampling_strategy='majority',random_state=43,voting='auto')
        X_res_down,y_res_down = cc.fit_resample(self.X,self.y)

        print('ClusterCentroid')
        self.print_result(y_res_down)

        return X_res_down,y_res_down       
    

#oversampling
from imblearn.over_sampling import ADASYN

class Oversampler:
    # initialize properties
    X=None
    y=None

    def __init__(self,X,y):
        self.X=X
        self.y=y

        print('OS Method: [ADASYN]')

    #display
    def print_result(self,y_after):
        print(f'After: {Counter(y_after)} \n')
    
    #ADASYN
    def ADASYN(self):
        ada = ADASYN(sampling_strategy='minority',random_state=43)
        X_res_up,y_res_up = ada.fit_resample(self.X,self.y)

        print('ADASYN')
        self.print_result(y_res_up)

        return  X_res_up,y_res_up