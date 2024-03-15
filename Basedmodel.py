from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Model:
    #result properties
    results = {}

    #init properties
    X_train,y_train = None,None
    X_test,y_test = None,None
    
    def __init__(self,data_train=None,data_test=None):
        if data_train and data_test is not None:            
            self.X_train, self.y_train = data_train
            self.X_test, self.y_test = data_test
            print(self.X_train.shape)
            print(self.y_train.shape)
            print(self.X_test.shape)
            print(self.y_test.shape)
        
    def result(self,model,y_test,y_pred):
        acc = accuracy_score(y_test,y_pred)
        cr = classification_report(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)

        acc_score = {model: acc}
        self.results.update(acc_score)
        print('Accuracy_Score  =',acc,'\n',cr,'\n',cm)

    def plot(self):
        sorted_results = dict(sorted(self.results.items(),key=lambda x:x[1]))

        keys = list(sorted_results.keys())
        values = list(sorted_results.values())

        ax = sns.barplot(x=keys,y=values,palette="rocket")

        for h, v in enumerate(values):
            ax.text(h, v/2, str(f'{v:.2f}'), ha='center', va='bottom', fontsize=10)

        plt.xlabel('Model')
        plt.ylabel('Accuracy Score')
        plt.title('Models Performance')

        plt.show()

    def Model(self,model = None):
        # k-nearest_neighbour
        if model == 'knn':
            clf = KNeighborsClassifier(n_neighbors=4,n_jobs=-1).fit(self.X_train,self.y_train)
            y_pred = clf.predict(self.X_test)

            self.result(model,self.y_test,y_pred)

        # naives_bayes
        elif model == 'nb':
            pass

        # logistic_regression
        elif model == 'lr':
            clf = LogisticRegression(random_state=43,solver='liblinear',multi_class='ovr',n_jobs=-1).fit(self.X_train,self.y_train)
            y_pred = clf.predict(self.X_test)

            self.result(model,self.y_test,y_pred)

        # support_vector_machine
        elif model == 'svm':
            clf = SVC(random_state=43).fit(self.X_train,self.y_train)
            y_pred = clf.predict(self.X_test)

            self.result(model,self.y_test,y_pred)
        
        elif model == 'rf':
            clf = RandomForestClassifier(random_state=43,n_jobs=-1).fit(self.X_train,self.y_train)
            y_pred = clf.predict(self.X_test)

            self.result(model,self.y_test,y_pred)

        # neural_network
        elif model == 'nn':
            X_train_nn,X_valid,y_train_nn,y_valid = train_test_split(self.X_train,self.y_train,random_state=43,stratify=self.y_train,test_size=0.25)
            print(y_train_nn.shape,y_valid.shape)

            input_shape = (self.X_train.shape[1],)
            clf = keras.Sequential()

            clf.add(layers.Dense(32,activation='relu', name='layer_1', input_shape=input_shape))
            clf.add(layers.Dense(64,activation='relu',name='layer_2'))
            clf.add(layers.Dense(64,activation='relu',name='layer_3'))
            clf.add(layers.Dense(32,activation='relu',name='layer_4'))
            clf.add(layers.Dense(4,activation='softmax',name='output_layer'))

            clf.summary()

            clf.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            
            clf.fit(x=X_train_nn,
                    y=y_train_nn,
                    batch_size=16,
                    epochs=30,
                    validation_data=(X_valid,y_valid)
                    )

            y_pred = np.argmax(clf.predict(self.X_test),axis=1)
            self.result(model,self.y_test,y_pred)

        else: 
            return 'No Model Found!'
