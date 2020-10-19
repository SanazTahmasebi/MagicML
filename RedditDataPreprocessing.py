import numpy as np
import json
import pdb
import cPickle as pickle

from operator import itemgetter

N = 10000               # Number of training data
Nv = 1000               # Number of validation data
Ns = 1000               # Number of test data
OffS = 0                # Offset to bypasss some frequent words
Nw = 160              # Number of frequent word features
p = False              #Enabling punctuation removal
ExFea = 5               # Number of extra features: 3 or 5
with open("proj1_data.json") as fp:
    data = json.load(fp)

####################################################################
# Parts 1, 2, 3, 4
####################################################################    
ss = ''       

for i in range(N):
    data_point = data[i]    
    ss += data_point['text'].lower()

# Extra Processing
if p==True:
    table = str.maketrans(",.!?()-\"*",9*" ")    
    ss=ss.translate(table)    

#pdb.set_trace()
Wc = dict()
ss2 = ss.split()


for word in ss2:
    if word in Wc:
        Wc[word]+=1
    else:
        Wc[word]=1
        
WWc = sorted(Wc.items(), key=itemgetter(1), reverse = True)      

# Most frequent words dictionary
topWc = WWc[OffS:OffS+Nw]  

#Initializing the feature vector
X = np.zeros((N,Nw+ExFea+1))
Y = np.zeros((N,1))

Xv = np.zeros((Nv,Nw+ExFea+1))
Yv = np.zeros((Nv,1))

Xs = np.zeros((Ns,Nw+ExFea+1))
Ys = np.zeros((Ns,1))

#Extracting Features
for i in range(N+Nv+Ns):
    data_point = data[i]    
    if i< N:        
        Y[i] = data_point['popularity_score']
        X[i][0]=1
        X[i][Nw+1]=data_point['is_root']
        X[i][Nw+2]=data_point['controversiality']
        X[i][Nw+3]=data_point['children']
        ss = data_point['text'].lower()
        if p==True:
            table = str.maketrans(",.!?()-\"*",9*" ")    
            ss=ss.translate(table)   
        ss2 = ss.split()
        if ExFea>3:
            X[i][Nw+4]=len(ss)
            X[i][Nw+5]=len(ss2)
        for j in range(Nw):           
            for word in ss2:
                if word==topWc[j][0]:
                    X[i][j+1]+=1                         
    elif (N<=i)and(i<N+Nv):
        k = i-N
        Yv[k] = data_point['popularity_score']
        Xv[k][0]=1
        Xv[k][Nw+1]=data_point['is_root']
        Xv[k][Nw+2]=data_point['controversiality']
        Xv[k][Nw+3]=data_point['children']
        ss = data_point['text'].lower()
        if p==True:
            table = str.maketrans(",.!?()-\"*",9*" ")    
            ss=ss.translate(table)   
        ss2 = ss.split()
        if ExFea>3:
            Xv[k][Nw+4]=len(ss)
            Xv[k][Nw+5]=len(ss2)
        for j in range(Nw):           
            for word in ss2:
                if word==topWc[j][0]:
                    Xv[k][j+1]+=1  
    else:
        k = i-N-Nv
        Ys[k] = data_point['popularity_score']
        Xs[k][0]=1
        Xs[k][Nw+1]=data_point['is_root']
        Xs[k][Nw+2]=data_point['controversiality']
        Xs[k][Nw+3]=data_point['children']
        ss = data_point['text'].lower()
        if p==True:
            table = str.maketrans(",.!?()-\"*",9*" ")    
            ss=ss.translate(table)   
        ss2 = ss.split()
        if ExFea>3:
            Xs[k][Nw+4]=len(ss)
            Xs[k][Nw+5]=len(ss2)
        for j in range(Nw):           
            for word in ss2:
                if word==topWc[j][0]:
                    Xs[k][j+1]+=1  
      
mi=np.zeros([Nw+ExFea,1])
si=np.zeros([Nw+ExFea,1])
    
for j in range(Nw+ExFea):
    mi[j]=np.mean(X[:,j+1])
    si[j]=np.std(X[:,j+1])    

for i in range(N+Nv+Ns):
    if i<N:
        for j in range(Nw+ExFea):            
            X[i,j+1]=(X[i,j+1]-mi[j])/si[j]
    elif (N<=i)and(i<N+Nv):
        k = i - N
        for j in range(Nw+ExFea):
            Xv[k,j+1]=(Xv[k,j+1]-mi[j])/si[j]
    else:
        k = i - N -Nv
        for j in range(Nw+ExFea):
            Xs[k,j+1]=(Xs[k,j+1]-mi[j])/si[j]
              
np.savez('FVec.npz', X=X, Y=Y)
np.savez('FVecV.npz', Xv=Xv, Yv=Yv)
np.savez('FVecS.npz', Xs=Xs, Ys=Ys)

     















