import numpy as np
import math

def testDriver():

    X,y = loadInData()
    
    lr = LogisiticRegression()
    lr.fit(X,y)



#Implementing Ridge Regression Logistic Regression
class LogisiticRegression:

    def __init__(self, lam = 10, max_iter1 = 30, max_iter2 = 200, e1 = .01, e2 = .005):
        self.l = lam
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2
        self.e1 = e1
        self.e2 = e2

    def fit(self,X,y):
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.y = y
        self.B = np.zeros([1,self.n])
        self.irls()
        
    #Predict given vector 
    def predict(self, x):
        return self.probability(x = x)
    
    #Returns log-likelyhood given the current state of Beta (B)
    #Solving sumOf(log(e^(y_i * x_i * B)  / (1 + e^(x_i * B)) ) - (lambda/2) * mag(B)^2)
    def logLikelyhood(self):
        res = 0
        for i in range(0,self.n):
            res += math.log(math.e^(self.y[i] * np.dot(self.X[i,:],self.B)) / (1 + math.e^(np.dot(self.X[i,:],self.B)))) - ((self.l/2) * np.linalg.norm(self.B)^2)
        return res
    
    #Get the deviance of the log likely hood
    def deviance(self):
        return -2 * self.logLikelyhood()
    
    #Return estimated probability of the ith sample from X
    def probability(self,x = None, idx = None):
        if(x):
            return 1 / (1 + math.e^(-1 * np.dot(self.X[idx,:], self.B)))
        elif(idx):
            return 1 / (1 + math.e^(-1 * np.dot(x, self.B)))
        else:
            return np.nan
        
    #LR w/ iteratively reweighted least squares 
    def irls(self):

        #Init values
        c = 0
        dev_c = 0
        dev_c1 = self.deviance()
        v = np.zeros([1,self.n])
        z = np.zeros([1,self.n])

        while abs((dev_c - dev_c1)/dev_c1) > self.e1 and c <= self.max_iter1:

            #Populate values for v and z
            for i in range(0,self.n):
                p_i = self.probability(idx = i) 
                v[i] = p_i(1-p_i)         
                z[i] = np.dot(self.X[i,:],self.B) + ((self.y[i] - p_i)/v[i]) #z_i = X_i * B + (y_i - p_i)/v_i

            #Generate V
            V = np.diag(v)

            #Update weights
            a = (np.linalg.multi_dot(np.transpose(self.X),V,self.X) + (self.l * self.X)) #A = X^T * V * X + Lambda * X
            b = (np.linalg.multi_dot(np.transpose(self.X),V,z)) #B = X * V * z
            self.B = self.cg(a,b)                               
            
            #Update deviance            
            dev_c = dev_c1 
            dev_c1 = self.deviance()
            c = c + 1

    #Conjugate Gradient to solve for weight
    def cg(self, a, b):
        r_c = b - np.dot(a,self.B)
        r_c1 = r_c
        c = 0
        direction_c = 1
        while np.linalg.norm(r_c1)^2 > self.e2 and c <= self.max_iter2:
            if c == 0:
                gamma = 0
            else:
                gamma = (np.dot(np.transpose(r_c1),(r_c1)))/(np.dot(np.transpose(r_c1),(r_c)))
            #Set direction
            direction_c1 = r_c1 + gamma*direction_c

            #Find Step
            s = np.dot(np.transpose(r_c),r_c)/np.linalg.multi_dot(np.transpose(direction_c),a,direction_c)

            #Update weights
            self.B = self.B + (gamma * direction_c1)

            #Find new residual
            r_c1 = r_c - np.dot(a, s, direction_c1)

            #Update values _c1 means "value for c + 1" and _c means current value
            r_c = r_c1
            direction_c = direction_c1
            c = c+1

    

