import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
root="/Users/adminadmin/Documents/mywork/e-commerce/yoochoose-data/"
def load_transaction(filepath):
    file = open(filepath, "r")
    clicklist = []
    userlist = []
    num_transaction = 0
    for line in file:
        num_transaction += 1
        L = line.split(" ")[1:]
        L[-1] = L[-1].replace("\n", "")
        clicklist.append(L)

        user = line.split(" ")[0]
        userlist.append(user)
    file.close()
    return clicklist,userlist,num_transaction
#load transaction
clicklist,userlist,num_transaction=load_transaction(root+"yoochoose-transaction-clicks.txt")
#load click sequence
clickseq,userlist,num_transaction=load_transaction(root+"yoochoose-sequence-clicks.txt")
#load bought users
buy_user=np.unique(np.loadtxt(root+"yoochoose-buys.dat",dtype=np.str, delimiter=",")[:,0])
print ("load data")

def initial_state(clickseq, clicklist): #real sequence, unique sequence
    size=len(clicklist)
    state=np.zeros(size,dtype=np.float)
    for i in range(size):
        target=clicklist[i]
        num=clickseq.count(target)
        state[i]=num
    distribution=state/len(clickseq)
    return distribution

#conditional probability
def conditional_probability(h,e):
    num_he=0
    num_e=0
    for transaction in clicklist:
        if h in transaction and e in transaction:
            num_he+=1
        if e in transaction:
            num_e+=1
    p= num_he/num_e
    return p

def transform_probability(e,h):
    num = 0
    for transaction in clicklist:
        for i in range(len(transaction) - 1):
            j = i + 1
            if transaction[i] == e and transaction[j] == h:
                num += 1
                break
    p = num / num_transaction
    return p

def sort_seprate(distribution,product):
    Z=zip(distribution,product)
    Z=sorted(Z,reverse=True)

    distribution1,product1=zip(*Z)
    cut=int(len(product1)/2)+1
    return product1[:cut]
'''
#Markov chain
file=open(root+"solution.txt","w")
num_user=0
for i in range(num_transaction):
    user=userlist[i]
    if user in buy_user:
        num_user+=1
        print (str(user)+" "+str(num_user)+"/"+str(len(buy_user)))
        buy_seq=clicklist[i]#unique item order
        # Markov matrix(buy_seq>=5), else output all
        size=len(buy_seq)
        if size<5:
            prediction=buy_seq
            print (prediction)
        else:
            graph = np.zeros((size, size), dtype=np.float)
            totaltime = size * size - size
            #count = 0
            for j in range(size):
                for k in range(size):
                    if j != k:
                        item1 = buy_seq[j]
                        item2 = buy_seq[k]
                        p_i2_i1 = transform_probability(item1, item2)
                        graph[j][k] = p_i2_i1
                        #count+=1
                        #print(str(count) + "/" + str(totaltime))
            #calculate diaginal value
            for j in range(size):
                line=graph[j]
                total=np.sum(line)
                graph[j][j]=1-total

            print (graph)
            #current state matrix
            #get click sequences
            clickseq1=clickseq[i]#real sequence
            initial_distribution=initial_state(clickseq1,buy_seq)
            next_distribution=np.dot(initial_distribution,graph)
            prediction=sort_seprate(next_distribution,buy_seq)
            print (prediction)

        file.write(user+" ")
        for l in range(len(prediction)):
            if l !=len(prediction)-1:
                file.write(prediction[l]+" ")
            else:
                file.write(prediction[l]+"\n")
        if num_user==1000:
            break
file.close()
'''
#test
predict_item,buy_user_list,num_buy=load_transaction(root+"solution.txt")
buy_item,buy_user_list1,num_buy1=load_transaction(root+"yoochoose-transaction.txt")
result=np.zeros((num_buy,2),dtype=np.float)
avgacc=0
for i in range(len(buy_user_list)):
    print (i)
    user=buy_user_list[i]
    pred=predict_item[i]

    index=np.argwhere(np.array(buy_user_list1)==user)[0][0]
    label=buy_item[index]

    intersect=set(pred)&set(label)
    union=set(pred)|set(label)
    accuracy=len(intersect)/len(label)
    avgacc=avgacc+accuracy
    result[i][0]=len(pred)
    result[i][1]=accuracy


avgacc=avgacc/num_buy
print ("avgacc: ",avgacc)

#resultdf=pd.DataFrame(result,columns=["length","accuracy"])
#resultdf.to_csv("/Users/adminadmin/Documents/mywork/e-commerce/recommend.csv")

plt.figure("histogram")
plt.hist(result[:,1],bins=3)
plt.xlabel("accuracy")
plt.ylabel("frequency")
plt.show()






