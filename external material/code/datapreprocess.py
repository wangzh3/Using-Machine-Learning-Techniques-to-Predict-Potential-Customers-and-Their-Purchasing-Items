import numpy as np
import datetime
import time
#root="/Users/adminadmin/Documents/mywork/e-commerce/"
def week(year,month, day):
    w = datetime.date(int(year), int(month), int(day))
    return w.weekday()+1

#user week1 hour1 week2 hour2 dw click item label
def preprocess_click(root,logpath,savepath):
    label = np.loadtxt(root+"yoochoose-buys.dat", dtype=np.str,
                       delimiter=",")
    print("load label")
    buyuser = np.unique(label[:, 0])
    all_sessions = np.loadtxt(logpath, dtype=np.str, delimiter=",")
    print (all_sessions.shape[0])
    stop=all_sessions.shape[0]-1
    file=open(savepath,"w")
    i=0
    while (i < all_sessions.shape[0]-1):
        product_list = []
        num_click=1
        print("head "+str(i)+"/"+str(all_sessions.shape[0]))
        user1=all_sessions[i,0] #start of the session

        file.write(user1 + " ")
        date=all_sessions[i,1].split("T")[0].split("-")
        year=date[0]
        month=date[1]
        day=date[2]
        week1 = str(week(year, month, day))
        time1=all_sessions[i,1].split("T")[1].split(":")
        hour1=time1[0]
        min1=time1[1]
        sec1=time1[2].replace("Z","")

        sec11=sec1.split(".")[0]
        sec12=sec1.split(".")[1]
        datetime1=datetime.datetime(int(year),int(month),int(day),int(hour1),int(min1),int(sec11),int(sec12))
        product = all_sessions[i, 2]
        product_list.append(product)
        for j in range(i+1,all_sessions.shape[0]):
            print(str(j) + "/" + str(all_sessions.shape[0]))
            user2=all_sessions[j,0]
            if(user1==user2):
                num_click+=1
                product = all_sessions[j, 2]
                product_list.append(product)
            else: # up is the end
                i=j
                index=j-1
                date = all_sessions[index, 1].split("T")[0].split("-")
                year = date[0]
                month = date[1]
                day = date[2]
                week2 = str(week(year, month, day))

                time1 = all_sessions[index, 1].split("T")[1].split(":")
                hour2 = time1[0]
                min2 = time1[1]
                sec2 = time1[2].replace("Z", "")

                sec21 = sec2.split(".")[0]
                sec22 = sec2.split(".")[1]
                datetime2 = datetime.datetime(int(year), int(month), int(day), int(hour2), int(min2), int(sec21),
                                              int(sec22))
                dwell=(datetime2-datetime1).seconds
                product_np = np.array(product_list)
                product_unique = np.unique(product_np)
                num_item=len(product_unique)
                list1=[str(week1),str(hour1),str(week2),str(hour2),str(dwell),str(num_click),str(num_item)]

                for item in list1:
                    file.write(item+" ")

                if (user1 in buyuser):
                    file.write("1\n")
                else:
                    file.write("0\n")

                break
            if (j == stop):
                i=stop
                index = j
                date = all_sessions[index, 1].split("T")[0].split("-")
                year = date[0]
                month = date[1]
                day = date[2]
                week2 = str(week(year, month, day))

                time1 = all_sessions[index, 1].split("T")[1].split(":")
                hour2 = time1[0]
                min2 = time1[1]
                sec2 = time1[2].replace("Z", "")

                sec21 = sec2.split(".")[0]
                sec22 = sec2.split(".")[1]
                datetime2 = datetime.datetime(int(year), int(month), int(day), int(hour2), int(min2), int(sec21),
                                              int(sec22))
                dwell = (datetime2 - datetime1).seconds

                product_np = np.array(product_list)
                product_unique = np.unique(product_np)
                num_item = len(product_unique)
                list1 = [str(week1), str(hour1), str(week2), str(hour2), str(dwell), str(num_click), str(num_item)]

                for item in list1:
                    file.write(item + " ")

                if (user1 in buyuser):
                    file.write("1\n")
                else:
                    file.write("0\n")

    file.close()

def preprocess_transaction(logpath,savepath):
    all_sessions = np.loadtxt(logpath, dtype=np.str, delimiter=",")
    print (all_sessions.shape[0])
    stop=all_sessions.shape[0]-1
    file=open(savepath,"w")
    i=0
    while (i < all_sessions.shape[0]-1):
        product_list=[]
        print("head "+str(i)+"/"+str(all_sessions.shape[0]))
        user1=all_sessions[i,0] #start of the session
        file.write(user1 + " ")
        product=all_sessions[i,2]
        product_list.append(product)

        for j in range(i+1,all_sessions.shape[0]):
            print(str(j) + "/" + str(all_sessions.shape[0]))
            user2=all_sessions[j,0]
            if(user1==user2):
                product=all_sessions[j,2]
                if product not in product_list:
                    product_list.append(product)

            else: # up is the end
                i=j
                for index in range(len(product_list)):
                    item=product_list[index]
                    if index != len(product_list)-1:
                        file.write(str(item) + " ")
                    else:
                        file.write(str(item)+"\n")
                break
            if (j == stop):
                i=stop
                for index in range(len(product_list)):
                    item = product_list[index]
                    if index != len(product_list) - 1:
                        file.write(str(item) + " ")
                    else:
                        file.write(str(item) + "\n")

    file.close()

def preprocess_sequence(logpath,savepath):
    all_sessions = np.loadtxt(logpath, dtype=np.str, delimiter=",")
    print (all_sessions.shape[0])
    stop=all_sessions.shape[0]-1
    file=open(savepath,"w")
    i=0
    while (i < all_sessions.shape[0]-1):
        product_list=[]
        print("head "+str(i)+"/"+str(all_sessions.shape[0]))
        user1=all_sessions[i,0] #start of the session
        file.write(user1 + " ")
        product=all_sessions[i,2]
        product_list.append(product)

        for j in range(i+1,all_sessions.shape[0]):
            print(str(j) + "/" + str(all_sessions.shape[0]))
            user2=all_sessions[j,0]
            if(user1==user2):
                product=all_sessions[j,2]
                product_list.append(product)

            else: # up is the end
                i=j
                for index in range(len(product_list)):
                    item=product_list[index]
                    if index != len(product_list)-1:
                        file.write(str(item) + " ")
                    else:
                        file.write(str(item)+"\n")
                break
            if (j == stop):
                i=stop
                for index in range(len(product_list)):
                    item = product_list[index]
                    if index != len(product_list) - 1:
                        file.write(str(item) + " ")
                    else:
                        file.write(str(item) + "\n")

    file.close()
'''
logpath=root+"yoochoose-clicks.dat"
savepath=root+"yoochoose-clickssave.txt" #9248730
preprocess_click(root,logpath,savepath)

dataset = np.loadtxt(savepath, dtype=np.float)
print ("load data")
#shuffle training dataset
np.random.shuffle(dataset)
file=open(root+"yoochoose-clickssave1.txt","w")
np.savetxt(root+"yoochoose-clickssave1.txt",dataset)
file.close()

#transcation
logpath=root+"yoochoose-buys.dat"
savepath=root+"yoochoose-transaction.txt" #9248730
preprocess_transaction(logpath,savepath)

logpath=root+"yoochoose-clicks.dat"
savepath=root+"yoochoose-transaction-clicks.txt"
preprocess_transaction(logpath,savepath)

#sequence
logpath=root+"yoochoose-data/yoochoose-clicks.dat"
savepath=root+"yoochoose-data/yoochoose-sequence-clicks.txt"
preprocess_sequence(logpath,savepath)
'''

