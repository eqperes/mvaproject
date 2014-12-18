
import numpy as np
import scipy.sparse
from scipy import linalg
from numpy import linalg as LA

#multi-view CCA
#parameters : 
#X = concatenation of all feature matrices
#index = [1,1,1,2,2] for example = 1 is for visual features, 2 for tag features (keeps the dimensions in mind)
#reg is for regularization, not necessary at first
#output :
#V=projection matrix
#D=diagonal matrix for projection
def CCA(X,index,reg) :
    C_all=np.matrix(np.cov(np.array(X.transpose())))
    C_diag=np.zeros(C_all.shape)
    print("done covariance matrix 1")
    print(index)
    for i in range(np.amax(index)) :
        indices=np.where(index==i+1)[0]
        for j in range(indices.shape[0]) : 
            index_f=indices[j]
            #add regularization here
            C_diag[index_f,index_f]=C_all[index_f,index_f]+reg*np.eye(C_all.shape[0],C_all.shape[1])[index_f,index_f]
            C_all[index_f,index_f]=C_all[index_f,index_f]+reg*np.eye(C_all.shape[0],C_all.shape[1])[index_f,index_f]
    print("done covariance matrix 2")
    print(C_all)
    print(C_diag)
    
    for i in range(C_all.shape[0]) :
        for j in range(C_all.shape[1]) :
            C_all[i,j]=float(C_all[i,j])
            C_diag[i,j]=float(C_diag[i,j])
            
    [D,V]=linalg.eig(C_all,C_diag)
    #print(D)
    #print(V)
    print("done eigen decomposition")
    a=-np.sort(-D,axis=0)#sort in descending order
    #print(a)
    index=np.argsort(-np.diag(D))[0,:]
    D=np.diag(a)
    #print(D)
    V=V[:,index]
    return[V,D]

def normalize(X) :
    for i in range(X.shape[0]) :
        X[1,:]=X[i,:]/norm(X[i,:])
    return X

#euclidian distance       
def dist_eucl(P1,P2) :
    P1=np.mat(P1)
    P2=np.mat(P2)
    dist_square=np.zeros(P1.shape[1])
    for i in range(P1.shape[1]) :
        dist_square[i]=(P1[0,i]-P2[0,i])*(P1[0,i]-P2[0,i])
    d=0
    for i in range(P1.shape[1]) :
        d=d+dist_square[i]
    return d

#CCA 2-views
#parameters :
#X=view 1 (eg result of neural networks for visual features)
#T=view 2 (eg result of LDA)
#output :
#Wx=projection matrix (concatenation of the projection matrices of the 2 features)
#to have the actual projection, do : P=[X,T]*Wx*D, and we can use P[:,index_of_feature] to retrieve the projection for the feature we want
def CCA2(X,T) :
    #T.todense()
    XX=np.concatenate((X,T),axis=1)#[X,T]
    index=np.concatenate((np.ones((X.shape[1],1),int),np.ones((T.shape[1],1),int)*2),axis=0)
    [V,D]=CCA(XX,index,0.0001)
    D=np.mat(D)
    Wx=V
    return [Wx,D]

#def CCA3(X,T,Y) :
#    T=full(T)
#    XX=[X,Y,T]
#    index=[np.ones((X.shape[1],1),float),np.ones((T.shape[1],1),float)*2,np.ones((Y.shape[1],1)),float)*3]
#    [V,D]=MultiviewCCA(XX,index,0.0001)
#    Wx=V
#    return [Wx,D]


#nearest neighbor :
#X=matrix of features for all images projected in the latent space (dim(X)=(nb of images,dimension of the latent space))
#target = vector of features in the latent space of the tag or image we want to find (dim(target)=(1,dimension of the latent space))
def NN(X,target) :
    dist=np.zeros(X.shape[0])
    for i in range(X.shape[0]) :
        dist[i]=dist_eucl(X[i,:],target)
    nearest_index=np.argsort(dist)[0]
    Nearest=X[nearest_index]
    return Nearest
