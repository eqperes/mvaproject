
import numpy as np
import scipy.sparse
from scipy import linalg
from numpy import linalg as LA

#multi-view CCA
def CCA(X,index,reg) :
    C_all=np.asmatrix(np.cov(np.asarray(X)))
    C_diag=np.zeros(C_all.shape)
    print("done covariance matrix 1")
    print(index)
    for i in range(1,np.amax(index)) :
        index_f=np.where(index==i)[0]
        #add regularization here
        C_diag[index_f,index_f]=C_all[index_f,index_f]+reg*np.eye(index_f.size,index_f.size)[index_f,index_f]
        C_all[index_f,index_f]=C_all[index_f,index_f]+reg*np.eye(index_f.size,index_f.size)[index_f,index_f]
    print("done covariance matrix 2")
    
    for i in range(1,C_all.shape[0]) :
        for j in range(1,C_all.shape[1]) :
            C_all[i,j]=float(C_all[i,j])
            C_diag[i,j]=float(C_diag[i,j])
    [V,D,s,t]=linalg.qz(C_all,C_diag)

    print("done eigen decomposition")
    a=-np.sort(-np.diag(D))#sort in descending order
    index=np.argsort(-np.diag(D))
    D=np.diag(a)
    V=V[:,index]
    return[V,D]

def normalize(X) :
    for i in range(1,X.shape[0]) :
        X[1,:]=X[i,:]/norm(X[i,:])
    return X

#euclidian distance       
def dist_pyth(P1,P2) :
    D = -2*P1*P2.conj().transpose()
    n1=np.zeros((1,P1.shape[0]))
    n2=np.zeros((1,P2.shape[0]))
    for i in range(0,P1.shape[0]-1) :
        n1[0,i]=LA.norm(P1[i,:])
    for i in range(0,P2.shape[0]-1) :
        n2[0,i]=LA.norm(P2[i,:])

    for i in range(0,P1.shape[0]-1) :
        for j in range(0,P2.shape[0]-1) :
            D[i,j]=D[i,j]+n1[0,i]*n1[0,i]+n2[0,j]*n2[0,j]
    for i in range(0,P1.shape[0]-1) :
        for j in range(0,P2.shape[0]-1) :
            D[i,j]=np.math.sqrt(D[i,j])
            
    return D

#CCA 2-views
def CCA2(X,T) :
    #T.todense()
    XX=np.concatenate((X,T),axis=1)#[X,T]
    index=np.concatenate((np.ones((X.shape[1],1),int),np.ones((T.shape[1],1),int)*2),axis=0)
    [V,D]=CCA(XX,index,0.0001)
    Wx=V
    return [Wx,D]

#def CCA3(X,T,Y) :
#    T=full(T)
#    XX=[X,Y,T]
#    index=[np.ones((X.shape[1],1),float),np.ones((T.shape[1],1),float)*2,np.ones((Y.shape[1],1)),float)*3]
#    [V,D]=MultiviewCCA(XX,index,0.0001)
#    Wx=V
#    return [Wx,D]

#basic nearest neighbor (to do image-to-image, tag-to-image and tag-to-tag retrieval)
def NN(P1,P2) :
    D=dist_pyth(P1,P2)
    Nearest=np.sort(D)[0]
    return Nearest
