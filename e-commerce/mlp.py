import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import metrics
import time
from matplotlib import pyplot as plt
#9249729 total
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden1=torch.nn.Linear(7,32)
        self.hidden2=torch.nn.Linear(32,32)
        self.hidden3 = torch.nn.Linear(32, 16)
        self.predict=torch.nn.Linear(16,2)

    def forward(self,x):
        x1=torch.nn.functional.tanh(self.hidden1(x))
        x2=torch.nn.functional.tanh(self.hidden2(x1))
        x3 = torch.nn.functional.tanh(self.hidden3(x2))
        output=torch.nn.functional.softmax(self.predict(x3))
        return output

class train_dataset(Dataset):
    def __init__(self,shuffledata):
        self.dataset = shuffledata[0:9229729,:]

        self.data = self.dataset[:,1:-1]
        self.tensordata = torch.from_numpy(self.data)

        self.label = self.dataset[:, -1]

        self.tensorlabel=torch.from_numpy(self.label).long()

    def __getitem__(self, i):
        x=self.tensordata[i]
        y=self.tensorlabel[i]
        return x,y

    def __len__(self):
        return self.data.shape[0]

class val_dataset(Dataset):
    def __init__(self, shuffledata):
        self.dataset = shuffledata[9229729:9239729, :]

        self.data = self.dataset[:,1:-1]
        self.tensordata = torch.from_numpy(self.data)

        self.label = self.dataset[:, -1]

        self.tensorlabel = torch.from_numpy(self.label).long()

    def __getitem__(self, i):
        x = self.tensordata[i]
        y = self.tensorlabel[i]
        return x, y

    def __len__(self):
        return self.data.shape[0]

class test_dataset(Dataset):
    def __init__(self, shuffledata):
        self.dataset = shuffledata[9239729:, :]

        self.data = self.dataset[:,1:-1]
        self.tensordata = torch.from_numpy(self.data)

        self.label = self.dataset[:, -1]

        self.tensorlabel = torch.from_numpy(self.label).long()

    def __getitem__(self, i):
        x = self.tensordata[i]
        y = self.tensorlabel[i]
        return x, y

    def __len__(self):
        return self.data.shape[0]

def recall(y,pred):
    # recall
    N = 0
    T = 0
    for item in y:
        if item == 0:
            N += 1
        else:
            T += 1
    class0 = 0
    class1 = 0
    for i in range(len(pred)):
        predict = pred[i]
        label = y[i]

        if predict==0 and label==0:
            class0+=1
        if predict==1 and label==1:
            class1+=1

    return class0/N, class1 / T

shuffledata1=np.loadtxt("/Users/adminadmin/Documents/mywork/e-commerce/yoochoose-data/yoochoose-clickssave1.txt",dtype=np.float)
print (shuffledata1.shape)

BS=50000
EP=100
train_data=train_dataset(shuffledata1)
train_loader=DataLoader(train_data,batch_size=BS,shuffle=False)

val_data=val_dataset(shuffledata1)
val_loader=DataLoader(val_data, batch_size=val_data.__len__(),shuffle=False)

test_data=test_dataset(shuffledata1)
test_loader=DataLoader(test_data, batch_size=test_data.__len__(),shuffle=False)
#train start
'''
net=Net()
net.double()

optimizer=torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func=torch.nn.CrossEntropyLoss()

time_per_ep=train_data.__len__()/BS

file = open("/Users/adminadmin/Documents/mywork/e-commerce/log.txt", "w")
for epoch in range(EP):
    time=1
    for x1,y1 in train_loader:
        predict1=net(x1)
        loss1=loss_func(predict1,y1)

        pred1=torch.max(predict1,1)[1]
        accuracy1=metrics.accuracy_score(y1.detach().numpy(),pred1.detach().numpy())
        recall1=metrics.confusion_matrix(y1.detach().numpy(),pred1.detach().numpy())
        recall10=recall1[1][0]
        recall11=recall1[1][1]
        recall_rate1=recall11/(recall10+recall11)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        print(str(epoch)+" "+str(time)+"/"+str(time_per_ep)+" "+str(loss1.item())+" "+str(accuracy1)+" "+str(recall_rate1))
        time+=1

    print("===========val===============")
    for x2,y2 in val_loader:
        predict2=net(x2)
        loss2=loss_func(predict2,y2)

        pred2 = torch.max(predict2, 1)[1]
        accuracy2 = metrics.accuracy_score(y2.detach().numpy(), pred2.detach().numpy())
        recall2 = metrics.confusion_matrix(y2.detach().numpy(), pred2.detach().numpy())
        recall20=recall2[1][0]
        recall21=recall2[1][1]
        recall_rate2=recall21/(recall20+recall21)

        print(str(epoch) + " " +str(loss2.item()) + " " + str(accuracy2) + " " + str(recall_rate2))
        file.write(str(epoch) + " " +str(loss2.item()) + " " + str(accuracy2) + " " + str(recall_rate2)+"\n")

    if(epoch%10==0):
        torch.save(net, "/Users/adminadmin/Documents/mywork/e-commerce/mlp/net" + str(epoch) + ".pkl")

torch.save(net, "/Users/adminadmin/Documents/mywork/e-commerce/mlp/net.pkl")
file.close()
#train end
'''
#test start
now = time.time()
net2=torch.load("/Users/adminadmin/Documents/mywork/e-commerce/mlp/net0.pkl")
loss_func=torch.nn.CrossEntropyLoss()

for x3,y3 in test_loader:
        predict3=net2(x3)
        loss3=loss_func(predict3,y3)
        print ("loss: ",loss3)
        pred3=torch.max(predict3,1)[1]
        accuracy3=metrics.accuracy_score(y3.detach().numpy(),pred3.detach().numpy())
        recall3=metrics.confusion_matrix(y3.detach().numpy(),pred3.detach().numpy())

        recall30 = recall3[1][0]
        recall31 = recall3[1][1]
        recall_rate3 = recall31 / (recall30 + recall31)
        print ("acc: ",accuracy3)
        print("recall: ",recall_rate3)
        print (recall3)
now1=  time.time()
print (now1-now)

'''
A=np.loadtxt("/Users/adminadmin/Documents/mywork/e-commerce/log.txt",dtype=np.float)
epoch=A[:,0]
loss=A[:,1]
acc=A[:,2]
plt.figure("loss")
plt.plot(epoch,loss)
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure("acc")
plt.plot(epoch,acc)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()
'''





