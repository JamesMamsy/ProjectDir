import numpy as np
import math


#Implementing Ridge Regression Logistic Regression
class LogisiticRegression:

    def __init__(self, a,iter, b0 = 1 ):
        self.max_iter = iter
        self.alpha = a
        self.b0 = b0

    def fit(self,X,y):
        self.X = X#self.normalize_by_column(X)  # n X m
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.y = y
        self.B = np.zeros(self.m)
        self.train()
        
        
    #Predict given vector 
    def predict(self, x):
        return self.probability(x = x) > 0.5
    
    
    #Returns log loss of model with current weights, probably won't need it for my lack of training
    def log_loss(self):
        y_hat = self.probability()
        y0_loss = self.y * y_hat #Loss for 0 cases
        y1_loss = (1 - self.y) * self.probability(1 - self.X)
        return -np.mean(y0_loss + y1_loss)

    
    #Return estimated probability of the itxth sample from X or from given X
    def probability(self,x = None, idx = None):
        if(idx != None):
            return self.probability(x=self.X[idx,:])
        elif(x is not None):
            return 1 / (1 + np.exp(-1 * np.dot(x, self.B) + self.b0)) 
        else:
            return self.probability(x=self.X)
        
    
    
    #Finds gradient of weights (B) to determine direction of change
    def gradient(self):
        diff = self.y - self.probability()
        grad_b = np.mean(diff)
        grad_w = self.X.T @ diff
        grad_w = np.array([np.mean(g) for g in grad_w])

        return grad_w,grad_b #Returns gradient of weights (B) and bias (B_0)
        

    def updateWeights(self):
        grad_w,grad_b = self.gradient()
        self.B = self.B - self.alpha * grad_w
        self.b0 = self.b0 * grad_b
    
    
    #Score predictions all given instances of X against y
    #Assumes numpy objects
    def accuracy_score(self,X,y):
        num_correct = 0
        for i,val in enumerate(X):
            if(self.predict(X[i]) == y[i]):
                num_correct += 1
        return num_correct/y.size

    #Min-Max normalization based on: https://stackoverflow.com/questions/27802109/column-wise-normalization-scaling-of-arrays
    def normalize_by_column(self,A):
        for col in range(A.shape[1]):
            A[:,col] = (A[:,col] - np.min(A[:,col])) / (np.max(A[:,col]) - np.min(A[:,col]))

   
    def train(self):

        for round in range(self.max_iter):
            self.updateWeights()
            print(self.b0)
            print(np.dot(self.X, self.B)[1])
            #print((1 + np.exp(-1 * np.dot(self.X, self.B) + self.b0))[1])
            print(self.log_loss())
    
    def prep_data(self,data):
        data = data[:100000].copy()

        top20_airport_ids = data["ORIGIN_AIRPORT_ID"].value_counts()[0:10].keys()

        data = data[data["ORIGIN_AIRPORT_ID"].isin(top20_airport_ids)]
        data.reset_index(drop=True, inplace=True)

        selected_collumns = ["ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","OP_UNIQUE_CARRIER","DEP_DELAY"]
        isDelayed = [1 if x > 0 else 0 for x in data['ARR_DELAY']]

        prepped_data = data[selected_collumns].copy()
        prepped_data["OP_UNIQUE_CARRIER"] = prepped_data["OP_UNIQUE_CARRIER"].apply(lambda x: int(x,36))
        prepped_data = prepped_data.to_numpy()

        prepped_data = np.insert(prepped_data,0,np.asarray(isDelayed),axis=1)
        prepped_data = prepped_data[~np.isnan(prepped_data).any(axis=1),:]
        X = prepped_data[:1000,1:]
        y = prepped_data[:1000,:1]
        return X,y
        
        

