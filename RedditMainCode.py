import numpy as np
import pdb
from matplotlib import pyplot

#pdb.set_trace()

def linear_regression_gradient(Train_data,Train_y,valid_x,valid_y,learn_rate=0.0001,decay=0,max_Iter=200):
    x=X
    y=Y
    WL=10000
    size_x,size_y=x.shape
    W=0.02*np.random.randn(size_y,1)
    #W=np.zeros([size_y,1])
    i=0
    mse=[]
    mse2=[]
    grad=[]
    while (i<max_Iter)and(WL>0.0001):
        alpha=learn_rate/(1+decay*(i+1))
        Wn=W-2*alpha*(np.dot(np.dot(x.transpose(),x),W)-np.dot(x.transpose(),y))
        grad.append(np.max(np.abs((np.dot(np.dot(x.transpose(),x),W)-np.dot(x.transpose(),y)))))
        i=i+1
        #        print(i)
        WL=np.sqrt(np.sum(np.square(Wn-W)))
        W=Wn
        mse.append((np.mean(np.square(y-np.dot(x,W)))))
        mse2.append((np.mean(np.square(Yv-np.dot(Xv,W)))))

    return W,mse,mse2

def linear_regression_closeform(X,Y,Xv,Yv):

    A=np.dot(np.transpose(X),X)
    #Ai=np.linalg.inv(A)
    Ai=np.linalg.pinv(A)
    WC=np.dot(np.dot(Ai,X.transpose()),Y)
    yc=np.dot(X,WC)
    mse=(np.mean(np.square(Y-yc)))
    ycv=np.dot(Xv,WC)
    mse2=(np.mean(np.square(Yv-ycv)))
    return WC,mse,mse2


F=np.load('FVec.npz')
F2=np.load('FVecS.npz')

X=F['X']
Y=F['Y']

Xv=F2['Xs']
Yv=F2['Ys']

Max_iter = 500
beta = 10/Max_iter

W,mse,mse2=linear_regression_gradient(X,Y,Xv,Yv,0.000002,beta,Max_iter)
WC,mse_c,mse_cv=linear_regression_closeform(X,Y,Xv,Yv)
pyplot.plot(mse)
pyplot.plot(mse2)
pyplot.legend(('Train MSE','Validation MSE'))
print('Closed form MSE is (Training): ',mse_c)
print('Gradient Descend MSE is (Training): ',mse[-1])
print('Closed form MSE is (Validation): ',mse_cv)
print('Gradient Descend MSE is (Validation): ',mse2[-1])