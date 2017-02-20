import time
import math
import numpy as np
from numba import vectorize, float32, float32, cuda
from scipy.special import digamma
import re
from scipy.sparse import dok_matrix
from scipy.special import gammaln

# device selection should be a cmd parameter
device = 0
user=int(1164556)
#the number of tweet
tweet=int(215037)
#the number of word
word=int(244848)
#the numeber of author
author=int(446014)
#the number of auxialry parameter
topic=int(50)
#the length of the retweet data in train.txt
length=int(16566756)
shp_off=0.1
rte_off=0.01
hp=0.3
btm=1
# Below are shape_next matrices transferred betweet host and device once in each iteration
u_shape_next = np.zeros((user,topic),dtype=np.float32) #cuda.pinned_array((500000,500))
v_shape_next = np.zeros((author,topic),dtype=np.float32)#cuda.pinned_array((5000,500))
y_shape_next = np.zeros((user,topic),dtype=np.float32)#cuda.pinned_array((500000,500))
theta_shape_next = np.zeros((tweet,topic),dtype=np.float32)#cuda.pinned_array((500000,500))
beta_shape_next = np.zeros((word,topic),dtype=np.float32)#cuda.pinned_array((500000,500))


elogu = cuda.pinned_array((user,topic),dtype=np.float32)
elogv = cuda.pinned_array((author,topic),dtype=np.float32)
elogy = cuda.pinned_array((user,topic),dtype=np.float32)
elogtheta = cuda.pinned_array((tweet,topic),dtype=np.float32)
elogbeta = cuda.pinned_array((word,topic),dtype=np.float32)

def computlog(a,b):
    return np.float32(digamma(a))-np.float32(np.log(b))

def random_offset(x,y,k):
    
    return np.random.rand(x,y)*k


@cuda.jit
def update_phis(ulist, alist, tlist, elbo, elu, elv, ely, eltheta, u_shape_n,
                v_shape_n, y_shape_n, theta_shape_n):
    #kk=50
    pos = cuda.grid(1)
    if pos < ulist.size:
        tempvec = cuda.local.array(2*50, dtype=float32)
        for j in range(50):
            tempvec[j] = elu[ulist[pos], j] + elv[alist[pos], j] + eltheta[tlist[pos],j]
            tempvec[50+j] = elu[ulist[pos], j] + ely[ulist[pos], j] + eltheta[tlist[pos],j]
        lsum = tempvec[0]
        for k in range(1, 2*50):
            if tempvec[k] < lsum:
                lsum = lsum + math.log(1 + math.exp(tempvec[k] - lsum))
            else:
                lsum = tempvec[k] + math.log(1 + math.exp(lsum - tempvec[k]))
        for k in range(2*50):
            tempvec[k] = math.exp(tempvec[k] - lsum)
        #phi=tempvec.copy_to_host()
        #print phi
        for k in range(50):
            ttmp = tempvec[k] + tempvec[k+50]
            temp1 = ttmp*(elu[ulist[pos], k] + eltheta[tlist[pos],k])
            temp2 = tempvec[k] * elv[alist[pos], k] + tempvec[k+50] * ely[ulist[pos], k]
            temp3 = tempvec[k] * math.log(tempvec[k]) + tempvec[k+50]*math.log(tempvec[k+50])
            elbo[pos]=elbo[pos] + temp1 + temp2 - temp3
            cuda.atomic.add(u_shape_n, (ulist[pos],k), ttmp)
            cuda.atomic.add(v_shape_n, (alist[pos],k), tempvec[k])
            cuda.atomic.add(y_shape_n, (ulist[pos],k), tempvec[k+50])
            cuda.atomic.add(theta_shape_n, (tlist[pos],k), ttmp)
    #return u_shape_next ,v_shape_next , y_shape_next,theta_shape_next
@cuda.jit
def update_phiz(tweetlist, wordlist, occur_num, elbo, eltheta, elbeta, theta_shape_n, beta_shape_n):
    pos = cuda.grid(1)
    if pos < tweetlist.size:
        tempvec = cuda.local.array(50, dtype=float32)
        for j in range(50):
            tempvec[j] = eltheta[tweetlist[pos], j] + elbeta[wordlist[pos], j]
        lsum = tempvec[0]
        for k in range(1,50):
            if tempvec[k] < lsum:
                lsum = lsum + math.log(1 + math.exp(tempvec[k] - lsum))
            else:
                lsum = tempvec[k] + math.log(1 + math.exp(lsum - tempvec[k]))
        for k in range(50):
            tempvec[k] = math.exp(tempvec[k] - lsum) * occur_num[pos]
            cuda.atomic.add(theta_shape_n, (tweetlist[pos],k), tempvec[k])
            cuda.atomic.add(beta_shape_n, (wordlist[pos],k), tempvec[k])
            temp1 = tempvec[k]*(eltheta[tweetlist[pos], k] + elbeta[wordlist[pos], k]) - tempvec[k]*math.log(tempvec[k])
            elbo[pos] =elbo[pos] + temp1
            #cuda.atomic.add(temp_phi,(pos,k),tempvec[k])

def updata_phis_single_threaded(userlis, authorlis, tweetlis, elu, elv,
                    ely, eltheta, u_shape_n, v_shape_n,
                    y_shape_n, theta_shape_n):
    tempvec = np.empty(2*elu.shape[1])
    for i in xrange(len(userlis)):
        for j in xrange(elu.shape[1]):
            tempvec[j] = elu[userlis[i], j] + elv[authorlis[i], j] + eltheta[tweetlis[i],j]
            tempvec[elu.shape[1]+j] = elu[userlis[i], j] + ely[userlis[i], j] + eltheta[tweetlis[i],j]
        lsum = tempvec[0]
        for k in xrange(1, 2*elu.shape[1]):
            if tempvec[k] < lsum:
                lsum = lsum + math.log(1 + math.exp(tempvec[k] - lsum))
            else:
                lsum = tempvec[k] + math.log(1 + math.exp(lsum - tempvec[k]))
        
        for k in xrange(2*elu.shape[1]):
            tempvec[k] = math.exp(tempvec[k] - lsum)
        for k in xrange(elu.shape[1]):
            u_shape_n[userlis[i],k] += tempvec[k] + tempvec[k+elu.shape[1]]
            v_shape_n[authorlis[i],k] += tempvec[k]
            y_shape_n[userlis[i],k] += tempvec[k+elu.shape[1]]
            theta_shape_n[tweetlis[i],k] += tempvec[k] + tempvec[k+elu.shape[1]]

def load_retweetdata():
    train=np.zeros((length,4))
    f=open('relation.txt','r')
    for index,line in enumerate(f):
        line=re.split(' ',line[:-2])
        #print index
        for num,value in enumerate(line):
            train[index,num] = int(value)
    f.close()
    return train

#load the tweet and word
def load_tweetdata():
    train=dok_matrix((tweet,word),dtype=np.float32)
    #content is the fotmat as : tweet index\n .word index
    f=open('content1.txt','r')
    for num,line in enumerate(f):
        if num%2==0:
            name=line.strip()
        if num%2==1:
            line=line.strip()
            value = re.split(' ',line)
            for item in value:
                train[int(name),int(item)]=train[int(name),int(item)]+1
                #pdb.set_trace()
    f.close()
    return train

def transfer(tweetdata):
    # this function is to get tweet and word index
    pos   = tweetdata.nonzero()
    times = tweetdata.values()
    length=len(tweetdata)
    result=np.zeros((length,3),dtype=np.int32)
    for i in range(length):
        result[i,0]=pos[0][i]
        result[i,1]=pos[1][i]
        result[i,2]=times[i]

    return result

def savemodel(X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE,iter):
    np.savetxt('model1/X_RTE_'+str(iter)+'.txt',X_RTE)
    np.savetxt('model1/U_SHP_'+str(iter)+'.txt',U_SHP)
    np.savetxt('model1/U_RTE_'+str(iter)+'.txt',U_RTE)
    np.savetxt('model1/Y_SHP_'+str(iter)+'.txt',Y_SHP)
    np.savetxt('model1/Y_RTE_'+str(iter)+'.txt',Y_RTE)
    np.savetxt('model1/V_SHP_'+str(iter)+'.txt',V_SHP)
    np.savetxt('model1/V_RTE_'+str(iter)+'.txt',V_RTE)
    np.savetxt('model1/THETA_SHP_'+str(iter)+'.txt',THETA_SHP)
    np.savetxt('model1/THETA_RTE_'+str(iter)+'.txt',THETA_RTE)
    np.savetxt('model1/BETA_SHP_'+str(iter)+'.txt',BETA_SHP)
    np.savetxt('model1/BETA_RTE_'+str(iter)+'.txt',BETA_RTE)


def  ELBO(hp,X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE, x_expec, u_expec, y_expec, v_expec, theta_expec, beta_expec):
    #after the update,we calculate the ELBO
    term1=np.sum((hp+topic*hp-X_SHP)*(digamma(X_SHP)-np.log(X_RTE))-(hp-X_RTE)*x_expec+gammaln(X_SHP)-X_SHP*np.log(X_RTE))
    term2=np.sum((hp-U_SHP)*(digamma(U_SHP)-np.log(U_RTE))-(np.reshape(x_expec,(user,1))-U_RTE)*u_expec+gammaln(U_SHP)-U_SHP*np.log(U_RTE))
    term3=np.sum((hp-Y_SHP)*(digamma(Y_SHP)-np.log(Y_RTE))-(hp-Y_RTE)*y_expec+gammaln(Y_SHP)-Y_SHP*np.log(Y_RTE))
    term4=np.sum((hp-V_SHP)*(digamma(V_SHP)-np.log(V_RTE))-(hp-V_RTE)*v_expec+gammaln(V_SHP)-V_SHP*np.log(V_RTE))
    term5=np.sum((hp-THETA_SHP)*(digamma(THETA_SHP)-np.log(THETA_RTE))-(hp-THETA_RTE)*theta_expec+gammaln(THETA_SHP)-THETA_SHP*np.log(THETA_RTE))
    term6=np.sum((hp-BETA_SHP)*(digamma(BETA_SHP)-np.log(BETA_RTE))-(hp-BETA_RTE)*beta_expec+gammaln(BETA_SHP)-BETA_SHP*np.log(BETA_RTE))
    term=term1+term2+term3+term4+term5+term6

    return term 

def make_nozero(a):
    a[np.where(a<np.float32(np.exp(-30)))]=np.float32(np.exp(-30))

    return a

if __name__ == "__main__":
    #    test
    prior = 0.3 # this should be different for different variables!
    dev = cuda.select_device(device)
    traindata=load_retweetdata()
    tweetdata=load_tweetdata()
    sparse_pos=transfer(tweetdata)
    X_SHP = np.zeros((1,user),dtype=np.float32)
    X_RTE = np.ones((1,user),dtype=np.float32)
    X_SHP = X_SHP+(hp+topic*hp)
    X_RTE = X_RTE*random_offset(1,user,rte_off)+hp

    U_SHP = np.ones((user,topic),dtype=np.float32)
    U_RTE = np.ones((user,topic),dtype=np.float32)
    U_SHP = U_SHP*random_offset(user,topic,shp_off)+hp
    U_RTE = U_RTE*random_offset(user,topic,rte_off)+hp 

    Y_SHP = np.ones((user,topic),dtype=np.float32)
    Y_RTE = np.ones((user,topic),dtype=np.float32)
    Y_SHP = Y_SHP*random_offset(user,topic,shp_off)+hp
    Y_RTE = Y_RTE*random_offset(user,topic,rte_off)+hp

    #author relevance
    V_SHP = np.ones((author,topic),dtype=np.float32)
    V_RTE = np.ones((1,topic),dtype=np.float32)
    V_SHP = V_SHP*random_offset(author,topic,shp_off)+hp
    V_RTE = V_RTE*random_offset(1,topic,rte_off)+hp
    
    #tweet relevance
    
    if btm==1:
        THETA_SHP=np.loadtxt('INIT_THETA_SHP.txt',dtype=np.float64)
        THETA_RTE=np.loadtxt('INIT_THETA_RTE.txt',dtype=np.float64)
        BETA_SHP =np.loadtxt('INIT_BETA_SHP.txt',dtype=np.float64)
        BETA_RTE = np.loadtxt('INIT_BETA_RTE.txt',dtype=np.float64)
        THETA_SHP= np.float32(THETA_SHP)
        THETA_RTE = np.float32(THETA_RTE)
        BETA_SHP= np.float32(BETA_SHP)
        BETA_RTE=np.float32(BETA_RTE)
    else:
        THETA_SHP = np.ones((tweet,topic),dtype=np.float32)
        THETA_RTE = np.ones((1,topic),dtype=np.float32) 
        THETA_SHP = THETA_SHP*hp
        THETA_RTE = THETA_RTE*hp

        BETA_SHP = np.ones((word,topic),dtype=np.float32)
        BETA_RTE = np.ones((1,topic),dtype=np.float32)
        BETA_SHP = BETA_SHP*hp
        BETA_RTE = BETA_RTE*hp

    ulist = np.int32(traindata[:,0])   
    alist = np.int32(traindata[:,1])
    tlist = np.int32(traindata[:,2])

    tweetlist = np.ascontiguousarray(np.int32(sparse_pos[:,0]))
    wordlist  = np.ascontiguousarray(np.int32(sparse_pos[:,1]))
    occur_num = np.ascontiguousarray(np.int32(sparse_pos[:,2]))

    #this is the data ,it's unchanged
    dev_ulist = cuda.to_device(ulist)
    dev_alist = cuda.to_device(alist)
    dev_tlist = cuda.to_device(tlist)

    dev_z_tlist = cuda.to_device(tweetlist)
    dev_z_wlist = cuda.to_device(wordlist)
    dev_z_olist = cuda.to_device(occur_num)

    theta_expec = make_nozero(THETA_SHP)/make_nozero(THETA_RTE)
    v_expec = make_nozero(V_SHP)/make_nozero(V_RTE)
    y_expec = make_nozero(Y_SHP)/make_nozero(Y_RTE)
    
    iter=0
    while 1:
        begain=time.time()
        #save some temp paraeter
        if iter%100==0 and iter>0:
            elbo_constant1=ELBO(hp,X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE, x_expec, u_expec, y_expec, v_expec, theta_expec, beta_expec)
            elbo_constant2=np.sum(np.sum(theta_expec,0)*np.sum(v_expec,0)*np.sum(u_expec,0))-author*np.sum(np.sum(theta_expec,0)*np.sum(u_expec*y_expec,0)) 
            elbo_constant3=np.sum(np.sum(theta_expec,0)*np.sum(beta_expec,0))
            elbo_constant4=(user+ user* topic+ author * topic + tweet * topic+ word* topic) * -1.4570 + user * topic* -1.0958
        #caluculate the current parameter's digamma derivation function 
        elogu[:,:]=computlog(U_SHP,U_RTE)
        elogv[:,:]=computlog(V_SHP,V_RTE)
        elogy[:,:]=computlog(Y_SHP,Y_RTE)
        elogtheta[:,:]=computlog(THETA_SHP,THETA_RTE)
        elogbeta[:,:]=computlog(BETA_SHP,BETA_RTE)
        #print type(elogu[1,1])
        #pdb.set_trace()
        #send to gpu
        elbo_init_s = np.zeros((length),dtype=np.float32)
        elbo_init_z = np.zeros((len(tweetlist)),dtype=np.float32)
        if iter==0:
            dev_elogu = cuda.to_device(elogu)
            dev_elogv = cuda.to_device(elogv)
            dev_elogy = cuda.to_device(elogy)
            dev_elogtheta = cuda.to_device(elogtheta)
            dev_elogbeta = cuda.to_device(elogbeta)
            dev_elbo_s = cuda.to_device(elbo_init_s)
            dev_elbo_z = cuda.to_device(elbo_init_z)
        else:
            dev_elogu.copy_to_device(elogu)
            dev_elogv.copy_to_device(elogv)
            dev_elogy.copy_to_device(elogy)
            dev_elogtheta.copy_to_device(elogtheta)
            dev_elogbeta.copy_to_device(elogbeta)
            dev_elbo_s.copy_to_device(elbo_init_s)
            dev_elbo_z.copy_to_device(elbo_init_z)
        u_shape_next[:,:] = prior
        v_shape_next[:,:] = prior
        y_shape_next[:,:] = prior
        theta_shape_next[:,:] = prior
        beta_shape_next[:,:] = prior

        #call cuda kernels to update all shape parameters
        threadsperblock = 256
        blockspergrid = (length + threadsperblock - 1) / threadsperblock
    
        time1 = time.time()
        update_phis[blockspergrid, threadsperblock](dev_ulist, dev_alist, dev_tlist,dev_elbo_s, 
                    dev_elogu, dev_elogv, dev_elogy, dev_elogtheta, u_shape_next, 
                    v_shape_next, y_shape_next, theta_shape_next)
        time2 = time.time()   
        #print "Cuda time for phi_s:" + str(time2-time1)
        elbo_s=dev_elbo_s.copy_to_host()

        threadsperblock = 256
        blockspergrid = (len(tweetlist) + threadsperblock - 1) / threadsperblock
        update_phiz[blockspergrid, threadsperblock](dev_z_tlist,dev_z_wlist,dev_z_olist,dev_elbo_z, 
                    dev_elogtheta, dev_elogbeta, theta_shape_next, beta_shape_next)
        elbo_z=dev_elbo_z.copy_to_host()
        #update shape parameters
        U_SHP = u_shape_next.copy()
        V_SHP = v_shape_next.copy()
        Y_SHP = y_shape_next.copy()
        THETA_SHP = theta_shape_next.copy()
        BETA_SHP = beta_shape_next.copy()

        #update U rate parameters
        u_part_1 = np.reshape(X_SHP/X_RTE,[user,1])
        u_part_2 = np.sum(theta_expec,0)*np.sum(v_expec,0)
        u_part_3 = np.sum(theta_expec,0)*author*y_expec
        U_RTE = u_part_1+u_part_2+u_part_3
        u_expec = make_nozero(U_SHP)/make_nozero(U_RTE)

        #update X rate parameters
        X_RTE = hp+np.sum(u_expec,1)
        x_expec=make_nozero(X_SHP)/make_nozero(X_RTE) 
        
        #update Y rate parameters
        Y_RTE = author*u_expec*np.sum(theta_expec,0)
        Y_RTE = Y_RTE + hp
        y_expec = make_nozero(Y_SHP)/make_nozero(Y_RTE)

        #update V rate parameters
        V_RTE = np.sum(u_expec,0)*np.sum(theta_expec,0) + hp
        v_expec = make_nozero(V_SHP)/make_nozero(V_RTE)

        #update Beta rate parameters
        BETA_RTE = np.sum(theta_expec,0) + hp
        beta_expec = make_nozero(BETA_SHP)/make_nozero(BETA_RTE)

        #update Theta rate parameters
        THETA_RTE = np.sum(u_expec,0)*np.sum(v_expec,0) + author*np.sum(u_expec*y_expec,0) + np.sum(beta_expec,0) + hp
        theta_expec = make_nozero(THETA_SHP)/make_nozero(THETA_RTE)
        
        #the end of the update
        end=time.time()
        print 'iteration time is',(end-begain)
        if iter%100==0 and iter>0:
            elbo=elbo_constant1 -elbo_constant2 - elbo_constant3 + elbo_constant4 +np.sum(elbo_s) + np.sum(elbo_z)
            print 'iter is  '+str(iter)
            print 'elbo is  '+str(elbo)
        if (iter)%50==0:
            savemodel(X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE,iter)        
        if iter==100:
            break
        iter=iter+1
        print iter
    
    cuda.close()
    
