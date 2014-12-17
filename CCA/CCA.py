
import numpy as np
import scipy.sparse
from scipy import linalg
from numpy import linalg as LA

#multi-view CCA
def CCA(X,index,reg) :
    C_all=np.matrix(np.cov(np.array(X.transpose())))
    C_diag=np.zeros(C_all.shape)
    print("done covariance matrix 1")
    print(index)
    for i in range(np.amax(index)) :
        index_f=np.where(index==i)[0]
        #add regularization here
        C_diag[index_f,index_f]=C_all[index_f,index_f]+reg*np.eye(index_f.size,index_f.size)[index_f,index_f]
        C_all[index_f,index_f]=C_all[index_f,index_f]+reg*np.eye(index_f.size,index_f.size)[index_f,index_f]
    print("done covariance matrix 2")
    print(C_all)
    print(C_diag)
    
    for i in range(C_all.shape[0]) :
        for j in range(C_all.shape[1]) :
            C_all[i,j]=float(C_all[i,j])
            C_diag[i,j]=float(C_diag[i,j])
            
    [D,V]=linalg.eig(C_all,C_diag)

    print("done eigen decomposition")
    a=-np.sort(-np.diag(D))#sort in descending order
    index=np.argsort(-np.diag(D))[0,:]
    D=np.diag(a)
    V=V[:,index]
    return[V,D]

def normalize(X) :
    for i in range(X.shape[0]) :
        X[1,:]=X[i,:]/norm(X[i,:])
    return X

#euclidian distance       
def dist_pyth(P1,P2) :
    P1=np.mat(P1)
    P2=np.mat(P2)
    D = -2*P1*P2.conj().transpose()
    #D=np.zeros(P1.shape[0])
    n1=np.zeros(P1.shape[0])
    n2=np.zeros(P2.shape[0])
    for i in range(P1.shape[0]) :
       n1[i]=LA.norm(P1[i])
    for i in range(P2.shape[0]) :
        n2[i]=LA.norm(P2[i])

    for i in range(P1.shape[0]) :
        for j in range(P2.shape[0]) :
            D[i,j]=D[i,j]+n1[i]*n1[i]+n2[j]*n2[j]#-2*n1[i]*n2[i]
    #dist=0
    for i in range(P1.shape[0]) :
        for j in range(P2.shape[0]) :
            D[i,j]=np.math.sqrt(D[i,j])
           
    return D

#CCA 2-views
def CCA2(X,T) :
    #T.todense()
    XX=np.concatenate((X,T),axis=1)#[X,T]
    index=np.concatenate((np.ones((X.shape[1],1),int),np.ones((T.shape[1],1),int)*2),axis=0)
    [V,D]=CCA(XX,index,0.0001)
    D=np.mat(D)
    Wx=V
 #   Wx=V[0,:]
#    for i in range(X.shape[1]-1) :
#        Wx=np.concatenate((Wx,V[i+1,:]))
#    Dx=[D[0]]
#    for i in range(X.shape[1]-1) :
#        Dx=np.concatenate((Dx,[D[i+1]]))
#    Projection_x=X*Wx*Dx
#    Wy=V[0,:]
#    for i in range(X.shape[1]-1) : 
#        Wy=np.concatenate((Wx,V[i+1+X.shape[1],:]))
#    Dy=[D[0]]
#    for i in range(X.shape[1]-1) :
#        Dy=np.concatenate((Dy,[D[i+1+X.shape[1]]]))
#    Projection_y=X*Wy*Dy
    Projection=XX*Wx*D.transpose()
    return Projection

#def CCA3(X,T,Y) :
#    T=full(T)
#    XX=[X,Y,T]
#    index=[np.ones((X.shape[1],1),float),np.ones((T.shape[1],1),float)*2,np.ones((Y.shape[1],1)),float)*3]
#    [V,D]=MultiviewCCA(XX,index,0.0001)
#    Wx=V
#    return [Wx,D]


#basic nearest neighbor (to do image-to-image, tag-to-image and tag-to-tag retrieval)
def NN(P1,P2) :
    #dist=np.zeros(Wx.shape[0])
    #Wx=np.sort(Wx)
    #for i in range(0,Wx.shape[0]-2) :
    #    dist[i]=dist_pyth(Wx[i,:],Wx[i+1,:])
    #ordered=np.sort(dist)
    #Nearest=dist[0]
    dist=dist_pyth(P1,P2)
    Nearest=np.sort(dist)[0]
    return Nearest
