import numpy as np
dataset = np.loadtxt("/Users/adminadmin/Documents/mywork/e-commerce/yoochoose-data/yoochoose-clickssave1.txt", dtype=np.float)
print ("load data")

B=dataset[:,-1]
print ("total")
print (len(B))
count0=0
count1=0
for item in B:
    if item==0:
        count0+=1
    if item==1:
        count1+=1

print("count0",count0)
print("count1",count1)

#train=B[0:9229729]
train=B[:1000000]
val=B[9229729:9239729]
test=B[9239729:]
count0=0
count1=0

for item in train:
    if item==0:
        count0+=1
    if item==1:
        count1+=1
print ("train")
print (len(train))
print("count0",count0)
print("count1",count1)

count0=0
count1=0
for item in val:
    if item==0:
        count0+=1
    if item==1:
        count1+=1
print ("val")
print (len(val))
print("count0",count0)
print("count1",count1)

count0=0
count1=0
for item in test:
    if item==0:
        count0+=1
    if item==1:
        count1+=1
print ("test")
print (len(test))
print("count0",count0)
print("count1",count1)

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

# how many items 19949(buys) 52739(clicks)
data=np.loadtxt(root+"yoochoose-buys.dat",dtype=np.str, delimiter=",")
product=data[:,2]
unique_product=np.unique(product)
print ("how many items: ",len(unique_product))

# number of each item pair
size=len(unique_product)
graph=np.zeros((size,size),dtype=np.float)
file=open(root+"yoochoose-transaction.txt","r")
num_transactions=0
maxlen=0
for line in file:
    num_transactions+=1
    L = line.split(" ")[1:]
    length=len(L)
    if length>maxlen:
        maxlen=length
    user=line.split(" ")[0]
    L[-1] = L[-1].replace("\n", "")
    for item1 in L:
        for item2 in L:
            print(user+" "+item1+" "+item2)
            index1=np.argwhere(unique_product==item1)[0][0]
            index2 = np.argwhere(unique_product == item2)[0][0]
            graph[index1][index2]+=1
            graph[index2][index1] += 1
print (graph)
#support value of each item pair
graph1=graph/float(num_transactions)
print (graph1)
file1=open(root+"unique_product.txt","w")
np.savetxt(file1,unique_product,fmt="%s")
file1.close()

file2=open(root+"transactiongraph.txt","w")
np.savetxt(file2,graph1)
file2.close()
#the longest buy sequence 53
print (maxlen)
file.close()

#distribution of 0: 0.99
matrix=np.loadtxt(root+"transactiongraph.txt")
num0=np.sum(matrix==0)
density=num0/(matrix.shape[0]*matrix.shape[1])
print (density)

