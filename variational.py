import scipy as sp
import re
import numpy as np
from scipy.sparse import dok_matrix
from scipy.special import digamma
from scipy.special import gamma
import pdb
from scipy.special import gammaln
#the number of user
user=int(4863)
#the number of tweet
tweet=int(99)
#the number of word
word=int(1667)
#the numeber of author
author=int(88)
#the number of auxialry parameter
topic=int(30)
#the length of the retweet data in train.txt
length=int(5205)
#the hyperparamenter we set,default is 0.3
hp=0.3
shp_off=0.1
rte_off=0.01
# this function is mean to give a offset to hyperparameter
BTM=0  #0 or 1
#initlize or not
def random_offset(x,y,k):
    return np.random.rand(x,y)*k


    #user relevance
    
#the phi matrix should be (user,author,tweet,topic),mow we reshape to (user*author*tweet,topic)
def update_phi_s(user_index,author_index,tweet_index,U_SHP,U_RTE,V_SHP,V_RTE,Y_SHP,Y_RTE,THETA_SHP,THETA_RTE):
    #phi_s1 is the multinomial parameter
    #Phi_s1 = dok_matrix((user*author*tweet,topic),dtype=np.float64)
    #Phi_s2 = dok_matrix((user*author*tweet,topic),dtype=np.float64)
    #calculate the index of the reshape matrix
    index = user_index + author_index*user + tweet_index*user*author
    index = int(index)
    U_RTE=make_nozero(U_RTE)
    THETA_RTE=make_nozero(THETA_RTE)
    V_RTE=make_nozero(V_RTE)
    Y_RTE=make_nozero(Y_RTE)
    temp = digamma(U_SHP[user_index,:]) - np.log(U_RTE[user_index,:]) + digamma(THETA_SHP[tweet_index,:]) - np.log(THETA_RTE[tweet_index,:])
    part_s1 = digamma(V_SHP[author_index,:]) - np.log(V_RTE[author_index,:])
    part_s2 = digamma(Y_SHP[user_index,:]) - np.log(Y_RTE[user_index,:])
    #phi_s1[index,:] = np.exp(part_s1+temp)
    #normalize
    temp1=part_s1+temp
    temp2=part_s2+temp
    total=np.ones((2*topic),dtype=np.float64)
    for i in range(topic):
        total[i]=temp1[i]
    for i in range(topic):
        total[i+topic]=temp2[i]
    #print len(total)
    normal=lognormalize(total)
    #print normal
    #print normal.shape
    phi_s1 = normal[0:topic]
    #phi_s1[index,:] = normal[0:topic]
    #phi_s1[index,:]=lognormalize(temp1)
    #phi_s2[index,:] = lognormalize(temp2)

    phi_s2 = normal[topic:2*topic]
    #phi_s2[index,:] = normal[topic:2*topic]
    #print np.sum(normal)
    return phi_s1,phi_s2

def update_phi_z(word_index,tweet_index,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE):
    #multimomial between word and tweet
    #phi_z = dok_matrix((word*tweet,topic),dtype=np.float64)
    index = word_index + tweet_index*word
    index = int(index)
    THETA_RTE=make_nozero(THETA_RTE)
    BETA_RTE=make_nozero(BETA_RTE)
    res = digamma(THETA_SHP[tweet_index,:]) - np.log(THETA_RTE[tweet_index,:]) + digamma(BETA_SHP[word_index,:]) - np.log(BETA_RTE[word_index,:])
    #phi_z[index,:] = np.exp(res)
    #normalize
    #print len(res)
    phi_z = lognormalize(res)
    #phi_z[index,:] = lognormalize(res)
    return phi_z
def logsum(a):
    for i in range(len(a)):
        if i==0:
            temp=a[i]
        else:
            if(a[i]<temp):
                temp=temp+np.log(1+np.exp(a[i]-temp))
            else:
                temp=a[i]+np.log(1+np.exp(temp-a[i]))
    return temp
def lognormalize(a):
    temp=logsum(a)
    for i in range(len(a)):
        a[i]=np.exp(a[i]-temp)
    #print a
    return a
#load the tain data,user,author,tweet,score
def load_retweetdata():
    train=np.zeros((length,4))
    f=open('relation.txt','r')
    for index,line in enumerate(f):
        line=re.split(' ',line[:-1])
        for num,value in enumerate(line):
            train[index,num] = int(value)
    f.close()
    return train

#load the tweet and word
def load_tweetdata():
    train={}
    #content is the fotmat as : tweet index\n .word index
    f=open('content.txt','r')
    for num,line in enumerate(f):
        if num%2==0:
            name=re.split('',line[:-1])
        if num%2==1:
            train[name[0]] = re.split(' ',line[:-2])
    f.close()
    return train

def loglikelyhood(X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE,length,traindata,tweet,tweetdata):
    X_expectation=(make_nozero(X_SHP)/make_nozero(X_RTE))
    #make_nozero(X_expectation)
    term1 = np.sum((hp-1)*np.log(X_expectation)-hp*X_expectation-np.log(gamma(hp))+hp*np.log(hp))
    U_expectation = (make_nozero(U_SHP)/make_nozero(U_RTE))
    #make_nozero(U_expectation)
    term2 = np.sum((hp-1)*np.log(U_expectation)-np.reshape(X_expectation,(user,1))*U_expectation-np.log(gamma(hp))+hp*np.log(np.reshape(X_expectation,(user,1))))
    Y_expectation=(make_nozero(Y_SHP)/make_nozero(Y_RTE))
    #make_nozero(Y_expectation)
    term3 = np.sum((hp-1)*np.log(Y_expectation)-hp*Y_expectation-np.log(gamma(hp))+hp*np.log(hp))
    V_expectation=make_nozero(V_SHP)/make_nozero(V_RTE)
    #make_nozero(V_expectation)
    term4 = np.sum((hp-1)*np.log(V_expectation)-hp*V_expectation-np.log(gamma(hp))+hp*np.log(hp))
    THETA_expectation=make_nozero(THETA_SHP)/make_nozero(THETA_RTE)
    #make_nozero(THETA_expectation)
    term5 = np.sum((hp-1)*np.log(THETA_expectation)-hp*THETA_expectation-np.log(gamma(hp))+hp*np.log(hp))
    #calculate predciton,possion disribution's expectation is his paramenter
    term6=0
    for i in range(length):
        s1 = U_expectation[traindata[i,0],:]*V_expectation[traindata[i,1],:]*THETA_expectation[traindata[i,2],:]
        s2 = U_expectation[traindata[i,0],:]*Y_expectation[traindata[i,0],:]*THETA_expectation[traindata[i,2],:]
        #make_nozero(s1)
        #make_nozero(s2)
        term6=term6+np.sum(s1*np.log(s1))-np.sum(np.log(s1))+np.sum(s2*np.log(s2))-np.sum(np.log(s2))
    term6=term6-np.sum(np.sum(THETA_expectation,0)*np.sum(V_expectation,0)*np.sum(U_expectation,0))-np.sum(np.sum(THETA_expectation,0)*np.sum(Y_expectation,0)*np.sum(U_expectation,0))
    BETA_expectation=make_nozero(BETA_SHP)/make_nozero(BETA_RTE)
    #make_nozero(BETA_expectation)
    term7=0
    for i in range(tweet):
        for j in tweetdata[str(i)]:
            #index = int(j)+i*word
            #print type(THETA_expectation)
            z=THETA_expectation[i,:]*BETA_expectation[int(j),:]
            term7=term7+np.sum(z*np.log(z))-np.sum(np.log(gamma(z+1)))
    term7=term7-np.sum(np.sum(THETA_expectation,0)*np.sum(BETA_expectation,0))
    term=term1+term2+term3+term4+term5+term5+term6+term7
    return term

def  ELBO(hp,X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE):
    #after the update,we calculate the ELBO
    U_RTE=make_nozero(U_RTE)
    THETA_RTE=make_nozero(THETA_RTE)
    V_RTE=make_nozero(V_RTE)
    Y_RTE=make_nozero(Y_RTE)
    THETA_RTE=make_nozero(THETA_RTE)
    BETA_RTE=make_nozero(BETA_RTE)
    X_SHP=make_nozero(X_SHP)
    U_SHP=make_nozero(U_SHP)
    Y_SHP=make_nozero(Y_SHP)
    V_SHP=make_nozero(V_SHP)
    THETA_SHP=make_nozero(THETA_SHP)
    BETA_SHP=make_nozero(BETA_SHP)
    BETA_RTE=make_nozero(BETA_RTE)
    X_expectation=make_nozero(X_SHP)/make_nozero(X_RTE)
    #make_nozero(X_expectation)
    term1=np.sum((hp+topic*hp-X_SHP)*(digamma(X_SHP)-np.log(X_RTE))-(hp-X_RTE)*X_expectation+gammaln(X_SHP)-X_SHP*np.log(X_RTE))
    U_expectation=make_nozero(U_SHP)/make_nozero(U_RTE)
    #make_nozero(U_expectation)
    term2=np.sum((hp-U_SHP)*(digamma(U_SHP)-np.log(U_RTE))-(np.reshape(X_expectation,(user,1))-U_RTE)*U_expectation+gammaln(U_SHP)-U_SHP*np.log(U_RTE))
    Y_expectation=make_nozero(Y_SHP)/make_nozero(Y_RTE)
    #make_nozero(Y_expectation)
    term3=np.sum((hp-Y_SHP)*(digamma(Y_SHP)-np.log(Y_RTE))-(hp-Y_RTE)*Y_expectation+gammaln(Y_SHP)-Y_SHP*np.log(Y_RTE))
    V_expectation=make_nozero(V_SHP)/make_nozero(V_RTE)
    #make_nozero(V_expectation)
    term4=np.sum((hp-V_SHP)*(digamma(V_SHP)-np.log(V_RTE))-(hp-V_RTE)*V_expectation+gammaln(V_SHP)-V_SHP*np.log(V_RTE))
    THETA_expectation=make_nozero(THETA_SHP)/make_nozero(THETA_RTE)
    #make_nozero(THETA_expectation)
    #print 'digamma '+str(np.sum((hp-THETA_SHP)*(digamma(THETA_SHP)-np.log(THETA_RTE))))
    #print 'digamma1  '+str(np.sum(-(hp-THETA_RTE)*THETA_expectation))
    #print 'digamma2  '+str(np.sum(np.log(gamma(THETA_SHP))-THETA_SHP*np.log(THETA_RTE)))
    term5=np.sum((hp-THETA_SHP)*(digamma(THETA_SHP)-np.log(THETA_RTE))-(hp-THETA_RTE)*THETA_expectation+gammaln(THETA_SHP)-THETA_SHP*np.log(THETA_RTE))
    BETA_expectation=make_nozero(BETA_SHP)/make_nozero(BETA_RTE)
    #make_nozero(BETA_expectation)
    term6=np.sum((hp-BETA_SHP)*(digamma(BETA_SHP)-np.log(BETA_RTE))-(hp-BETA_RTE)*BETA_expectation+gammaln(BETA_SHP)-BETA_SHP*np.log(BETA_RTE))
    #term7=0
    #for i in range(length):
    #    index = traindata[i,0] + traindata[i,1]*user + traindata[i,2]*user*author
    #    index = int(index)
        #s1 = U_expectation[traindata[i,0],:]*V_expectation[traindata[i,1],:]*THETA_expectation[traindata[i,2],:]
        #s2 = U_expectation[traindata[i,0],:]*Y_expectation[traindata[i,0],:]*THETA_expectation[traindata[i,2],:]
        #make_nozero(s1)
        #make_nozero(s2)
        #print s1.dtype
    #    temp1=np.zeros((1,topic),dtype=np.float64)
    #    temp2=np.zeros((1,topic),dtype=np.float64)
    #   for j in range(topic):
    #      temp1[0,j]=phi_s1[index,j]
    #        temp2[0,j]=phi_s2[index,j]
    #    temp=temp1*(digamma(U_SHP[traindata[i,0],:])-np.log(U_RTE[traindata[i,0],:])+digamma(V_SHP[traindata[i,1],:])-np.log(V_RTE[traindata[i,1],:])+digamma(THETA_SHP[traindata[i,2],:])-np.log(THETA_RTE[traindata[i,2],:]))+\
    #            temp2*(digamma(U_SHP[traindata[i,0],:])-np.log(U_RTE[traindata[i,0],:])+digamma(THETA_SHP[traindata[i,2],:])-np.log(THETA_RTE[traindata[i,2],:])+digamma(Y_SHP[traindata[i,0],:])-np.log(Y_RTE[traindata[i,0],:]))\
    #                -temp1*np.log(temp1)-temp2*np.log(temp2)
    #    term7=term7+np.sum(temp)
    #term7=term7-np.sum(np.sum(THETA_expectation,0)*np.sum(V_expectation,0)*np.sum(U_expectation,0))-author*np.sum(np.sum(THETA_expectation,0)*np.sum(U_expectation*Y_expectation,0))
    #term8=0
    #for i in range(tweet):
    #    for j in tweetdata[str(i)]:
    #        index = int(j)+i*word
    #        #z=THETA_expectation[i,:]*BETA_expectation[int(j),:]
    #        #make_nozero(z)
    #3        temp_z=np.zeros((1,topic),dtype=np.float64)
    #        for n in range(topic):
    #            temp_z[0,n]=phi_z[index,n]
    #        temp=temp_z*(digamma(THETA_SHP[i,:])-np.log(THETA_RTE[i,:])+digamma(BETA_SHP[int(j),:])-np.log(BETA_RTE[int(j),:]))-temp_z*np.log(temp_z)
    #        term8=term8+np.sum(temp)
    #term8=term8-np.sum(np.sum(THETA_expectation,0)*np.sum(BETA_expectation,0))
    #term=term1+term2+term3+term4+term5+term6+term7+term8+(user+ user* topic+ author * topic + tweet * topic+ word* topic) * -1.4570 + user * topic* -1.0958
    term=term1+term2+term3+term4+term5+term6
    print term1,term2,term3,term4,term5,term6
    return term

def predict_score(U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,traindata,length):
    prediction = traindata
    
    U_expectation = U_SHP/U_RTE
    THETA_expectation = THETA_SHP / THETA_RTE
    Y_expectation = Y_SHP/Y_RTE
    V_expectation = V_SHP/V_RTE
    for i in range(length):
        user_index = int(traindata[i,0])
        author_index = int(traindata[i,1])
        tweet_index = int(traindata[i,2])
        s1 = U_expectation[user_index,:]*V_expectation[author_index,:]*THETA_expectation[tweet_index,:]
        s2 = U_expectation[user_index,:]*Y_expectation[author_index,:]*THETA_expectation[tweet_index,:]
        #print type(prediction)
        #possion distribution's expectation is its parameter
        temp=normalize_sacalr(np.sum(s1)+np.sum(s2))
        #print type(temp)
        prediction[i,3]=float(temp)
    return s1,s2,prediction
def normalize_sacalr(score):
    if score<np.exp(-30):
        score=np.exp(-30)
    result=np.exp(-score)
    return 1-result


def normalize(score):
    # ensure the score between [0,1]
    score[np.where(score<np.exp(-30))]=np.exp(-30)
    result=np.exp(-score)
    
    return 1-result

def BTM_initialize(THETA_SHP,BETA_SHP):
    #use Btm initialize theta and beta
    THETA_expectation=np.zeros((tweet,topic))
    f=open('document.txt','r')
    for num,line in enumerate(f):
        line=re.split(' ',line[:-1])
        for value in range(len(line)):
            THETA_expectation[num,value]=float(line[value])+0.1
    THETA_expectation=THETA_expectation / np.reshape(np.sum(THETA_expectation,1),(document,1))
    f.close()
    THETA_RTE=make_nozero(THETA_SHP/THETA_expectation)

    f=open('word.txt','r')
    BETA_expectation=np.zeros((word,topic))
    for num,line in enumerate(f):
        line=re.split(' ',line[:-1])
        for value in range(len(line)):
            BETA_expectation[num,value]=float(line[value])+0.01
    BETA_expectation=BETA_expectation/ np.reshape(np.sum(BETA_expectation,1),(word,1))
    f.close()
    BETA_RTE = make_nozero(BETA_SHP/BETA_expectation)
    return THETA_RTE,BETA_RTE

def make_nozero(a):
    a[np.where(a<np.exp(-30))]=np.exp(-30)

    return a

def savemodel(X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE):
    np.savetxt('model1/X_SHP.txt',X_SHP)
    np.savetxt('model1/X_RTE.txt',X_RTE)
    np.savetxt('model1/U_SHP.txt',U_SHP)
    np.savetxt('model1/U_RTE.txt',U_RTE)
    np.savetxt('model1/Y_SHP.txt',Y_SHP)
    np.savetxt('model1/Y_RTE.txt',Y_RTE)
    np.savetxt('model1/V_SHP.txt',V_SHP)
    np.savetxt('model1/V_RTE.txt',V_RTE)      
    np.savetxt('model1/THETA_SHP.txt',THETA_SHP)
    np.savetxt('model1/THETA_RTE.txt',THETA_RTE)
    np.savetxt('model1/BETA_SHP.txt',BETA_SHP)
    np.savetxt('model1/BETA_RTE.txt',BETA_RTE)   

def process(user,tweet,word,author,topic,hp,shp_off,rte_off,BTM,length):
    X_SHP = np.ones((1,user),dtype=np.float64)
    X_RTE = np.ones((1,user),dtype=np.float64)
    X_SHP = X_SHP*random_offset(1,user,shp_off)+hp+topic*hp
    X_RTE = X_RTE*random_offset(1,user,rte_off)+hp

    U_SHP = np.ones((user,topic),dtype=np.float64)
    U_RTE = np.ones((user,topic),dtype=np.float64)
    U_SHP = U_SHP*random_offset(user,topic,shp_off)+hp
    U_RTE = U_RTE*random_offset(user,topic,rte_off)+hp 

    Y_SHP = np.ones((user,topic),dtype=np.float64)
    Y_RTE = np.ones((user,topic),dtype=np.float64)
    Y_SHP = Y_SHP*random_offset(user,topic,shp_off)+hp
    Y_SHP = Y_SHP*random_offset(user,topic,rte_off)+hp

    #author relevance
    V_SHP = np.ones((author,topic),dtype=np.float64)
    V_RTE = np.ones((author,topic),dtype=np.float64)
    V_SHP = V_SHP*random_offset(author,topic,shp_off)+hp
    V_RTE = V_RTE*random_offset(author,topic,rte_off)+hp

    #tweet relevance
    THETA_SHP = np.ones((tweet,topic),dtype=np.float64)
    THETA_RTE = np.ones((tweet,topic),dtype=np.float64) 
    THETA_SHP = THETA_SHP*random_offset(tweet,topic,shp_off)+hp
    THETA_RTE = THETA_RTE*random_offset(tweet,topic,rte_off)+hp

    BETA_SHP = np.ones((word,topic),dtype=np.float64)
    BETA_RTE = np.ones((word,topic),dtype=np.float64)
    BETA_SHP = BETA_SHP*random_offset(word,topic,shp_off)+hp
    BETA_RTE = BETA_RTE*random_offset(word,topic,rte_off)+hp
    #traindata is numpy[length,4], tweetdata is dictionanry,tweet id should be (0,1,2,3,......)

    traindata=load_retweetdata()
    tweetdata=load_tweetdata()
    #phi_s1 = dok_matrix((user*author*tweet,topic),dtype=np.float64)
    #phi_s2 = dok_matrix((user*author*tweet,topic),dtype=np.float64)
    #phi_z = dok_matrix((word*tweet,topic),dtype=np.float64)
    
    #initialize all the matix 
    res=open('result.txt','w')
    if BTM:
        BTM_initialize    

    iter=0
    while 1:
        #save the parameter in last iter
        X_SHP_TEMP=X_SHP
        X_RTE_TEMP=X_RTE
        U_SHP_TEMP=U_SHP
        U_RTE_TEMP=U_RTE
        Y_SHP_TEMP=Y_SHP
        Y_RTE_TEMP=Y_RTE
        V_SHP_TEMP=V_SHP
        V_RTE_TEMP=V_RTE
        THETA_SHP_TEMP=THETA_SHP
        THETA_RTE_TEMP=THETA_RTE
        BETA_SHP_TEMP=BETA_SHP
        BETA_RTE_TEMP=BETA_RTE

        X_SHP = np.ones((1,user),dtype=np.float64)
        X_SHP = X_SHP*random_offset(1,user,shp_off)+hp+topic*hp
        X_RTE = hp+np.sum(U_SHP/U_RTE,1)

        u_part_1 = np.reshape(X_SHP/X_RTE,[user,1])
        u_part_2 = np.sum(THETA_SHP_TEMP/THETA_RTE_TEMP,0)*np.sum(V_SHP_TEMP/V_RTE_TEMP,0)
        u_part_3 = np.sum(THETA_SHP_TEMP/THETA_RTE_TEMP,0)*author*(make_nozero(Y_SHP_TEMP)/make_nozero(Y_RTE_TEMP))
        #update U_RTE
        U_RTE = u_part_1+u_part_2+u_part_3
        #U_RTE = np.reshape(X_SHP/X_RTE,[user,1])+np.sum(THETA_SHP_TEMP/THETA_RTE_TEMP,0)*np.sum(V_SHP_TEMP/V_RTE_TEMP,0)
        #update Y_RTE
        #Y_RTE=np.ones((user,topic),dtype=np.float64)*hp
        #Y_RTE=Y_RTE+author*(U_SHP/U_RTE)*np.sum(THETA_SHP/THETA_RTE,0)
        #update the V_RTE
        #V_RTE = hp*np.ones((author,topic),dtype=np.float64) + np.sum(U_SHP/U_RTE,0)*np.sum(THETA_SHP/THETA_RTE,0)
        #save the temp theta
        #initial the hp
        U_SHP=np.ones((user,topic),dtype=np.float64)*hp
        Y_SHP=np.ones((user,topic),dtype=np.float64)*hp
        V_SHP=np.ones((author,topic),dtype=np.float64)*hp
        THETA_SHP=np.ones((tweet,topic),dtype=np.float64)*hp
        for i in range(length):
            phi_s1,phi_s2 = update_phi_s(traindata[i,0],traindata[i,1],traindata[i,2],U_SHP_TEMP,U_RTE_TEMP,V_SHP_TEMP,V_RTE_TEMP,Y_SHP_TEMP,Y_RTE_TEMP,THETA_SHP_TEMP,THETA_RTE_TEMP)
            index = traindata[i,0] + traindata[i,1]*user + traindata[i,2]*user*author
            index = int(index)
            U_SHP[traindata[i,0],:] = U_SHP[traindata[i,0],:] + phi_s1+phi_s2
            Y_SHP[traindata[i,0],:] = Y_SHP[traindata[i,0],:] + phi_s2
            THETA_SHP[traindata[i,2],:] = THETA_SHP[traindata[i,2],:] + phi_s1+ phi_s2
            V_SHP[traindata[i,1],:] = V_SHP[traindata[i,1],:] + phi_s1
        
        Y_RTE=np.ones((user,topic),dtype=np.float64)*hp
        Y_RTE=Y_RTE+author*(make_nozero(U_SHP)/make_nozero(U_RTE))*np.sum(make_nozero(THETA_SHP_TEMP)/make_nozero(THETA_RTE_TEMP),0)
        V_RTE = hp*np.ones((author,topic),dtype=np.float64) + np.sum(U_SHP/U_RTE,0)*np.sum(THETA_SHP_TEMP/THETA_RTE_TEMP,0)
        #calculate the t_part_3
        #t_part_3 = np.sum(BETA_SHP/BETA_RTE,0)

        BETA_SHP=np.ones((word,topic),dtype=np.float64)*hp
        for i in range(tweet):
            for j in tweetdata[str(i)]:
                phi_z = update_phi_z(int(j),i,THETA_SHP_TEMP,THETA_RTE_TEMP,BETA_SHP_TEMP,BETA_RTE_TEMP)
                #index = int(j)+i*word
                #index=int(index)
                BETA_SHP[int(j),:] = BETA_SHP[int(j),:]+phi_z
                THETA_SHP[i,:] = THETA_SHP[i,:]+phi_z

        t_part_3 = np.sum(BETA_SHP_TEMP/BETA_RTE_TEMP,0)
        t_part_1 = np.sum(U_SHP/U_RTE,0)*np.sum(V_SHP/V_RTE,0)
        t_part_2 = author*np.sum((U_SHP/U_RTE)*(Y_SHP/Y_RTE),0)
        THETA_RTE = hp*np.ones((tweet,topic),dtype=np.float64)+t_part_1+t_part_2+t_part_3
        BETA_RTE = hp*np.ones((word,topic),dtype=np.float64)+np.sum(THETA_SHP/ THETA_RTE,0)
        #every 10 times claculate the elbo
        #elbo=ELBO_origin(hp,X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE,length,traindata,tweet,tweetdata,phi_s1,phi_s2,phi_z)
        #print 'elbo is  '+str(elbo)
        #print 'iter is  '+str(iter)
        if iter%1==0:
            U_expectation=make_nozero(U_SHP)/make_nozero(U_RTE)
            X_expectation=make_nozero(X_SHP)/make_nozero(X_RTE)
            V_expectation=make_nozero(V_SHP)/make_nozero(V_RTE)
            Y_expectation=make_nozero(Y_SHP)/make_nozero(Y_RTE)
            THETA_expectation=make_nozero(THETA_SHP)/make_nozero(THETA_RTE)
            BETA_expectation=make_nozero(BETA_SHP)/make_nozero(BETA_RTE)
            #elbo=elbo+ELBO(hp,X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE)

            term7=0
            for i in range(length):
                phi_s1,phi_s2 = update_phi_s(traindata[i,0],traindata[i,1],traindata[i,2],U_SHP_TEMP,U_RTE_TEMP,V_SHP_TEMP,V_RTE_TEMP,Y_SHP_TEMP,Y_RTE_TEMP,THETA_SHP_TEMP,THETA_RTE_TEMP)
                temp=phi_s1*(digamma(U_SHP[traindata[i,0],:])-np.log(U_RTE[traindata[i,0],:])+digamma(V_SHP[traindata[i,1],:])-np.log(V_RTE[traindata[i,1],:])+digamma(THETA_SHP[traindata[i,2],:])-np.log(THETA_RTE[traindata[i,2],:]))+\
                        phi_s2*(digamma(U_SHP[traindata[i,0],:])-np.log(U_RTE[traindata[i,0],:])+digamma(THETA_SHP[traindata[i,2],:])-np.log(THETA_RTE[traindata[i,2],:])+digamma(Y_SHP[traindata[i,0],:])-np.log(Y_RTE[traindata[i,0],:]))\
                            -phi_s1*np.log(phi_s1)-phi_s2*np.log(phi_s2)
                term7=term7+np.sum(temp)
            term7=term7-np.sum(np.sum(THETA_expectation,0)*np.sum(V_expectation,0)*np.sum(U_expectation,0))-author*np.sum(np.sum(THETA_expectation,0)*np.sum(U_expectation*Y_expectation,0))
            term8=0
            for i in range(tweet):
                for j in tweetdata[str(i)]:
                    phi_z = update_phi_z(int(j),i,THETA_SHP_TEMP,THETA_RTE_TEMP,BETA_SHP_TEMP,BETA_RTE_TEMP)
                    temp=phi_z*(digamma(THETA_SHP[i,:])-np.log(THETA_RTE[i,:])+digamma(BETA_SHP[int(j),:])-np.log(BETA_RTE[int(j),:]))-phi_z*np.log(phi_z)
                    term8=term8+np.sum(temp)
            term8=term8-np.sum(np.sum(THETA_expectation,0)*np.sum(BETA_expectation,0))
            elbo=term7+term8+(user+ user* topic+ author * topic + tweet * topic+ word* topic) * -1.4570 + user * topic* -1.0958
           
            elbo=elbo+ELBO(hp,X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE)
            print 'iter is  '+str(iter)
            print 'elbo is  '+str(elbo)
            res.write('elbo '+str(elbo)+'\n'+'iter '+str(iter)+'\n')
        iter=iter+1
        if iter==100:
            break
        if iter%100==0:
            s1,s2,pre=predict_score(U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,traindata,length)
            np.savetxt(str(iter)+'pre.txt',pre)
    
    savemodel(X_SHP,X_RTE,U_SHP,U_RTE,Y_SHP,Y_RTE,V_SHP,V_RTE,THETA_SHP,THETA_RTE,BETA_SHP,BETA_RTE)


if __name__=='__main__':
  process(user,tweet,word,author,topic,hp,shp_off,rte_off,BTM,length)  

        

        




        



































