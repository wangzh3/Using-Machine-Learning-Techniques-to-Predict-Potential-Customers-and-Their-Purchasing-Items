import numpy as np
import datapreprocess
from sklearn import metrics, neighbors

root=input("Please input the folder path that datasets have (end with /:)")

while (True):
    print("Please input the one of the command below:")
    print("1. preprocess")
    print("2. analysis")
    cmd=input("Please input:")
    if(cmd=="preprocess"):
        print ("=====data pre-process for click events=====")
        logpath = root + "yoochoose-clicks.dat"
        savepath = root + "yoochoose-clickssave.txt"  # 9248730
        datapreprocess.preprocess_click(root,logpath, savepath)

        dataset = np.loadtxt(savepath, dtype=np.float)
        print("=====load data and shuffle=====")
        # shuffle training dataset
        np.random.shuffle(dataset)
        file = open(root + "yoochoose-clickssave1.txt", "w")
        np.savetxt(root + "yoochoose-clickssave1.txt", dataset)
        file.close()

        # transcation
        print("=====data pre-process for only buyer searched items=====")
        logpath = root + "yoochoose-buys.dat"
        savepath = root + "yoochoose-transaction.txt"  # 9248730
        datapreprocess.preprocess_transaction(logpath, savepath)

        print("=====data pre-process for all user searched items=====")
        logpath = root + "yoochoose-clicks.dat"
        savepath = root + "yoochoose-transaction-clicks.txt"
        datapreprocess.preprocess_transaction(logpath, savepath)

        # sequence
        print("=====data pre-process for all user searched sequences=====")
        logpath = root + "yoochoose-clicks.dat"
        savepath = root + "yoochoose-sequence-clicks.txt"
        datapreprocess.preprocess_sequence(logpath, savepath)

    elif(cmd=="analysis"):
        # select potential users
        print("Loading training data...")
        clicks = np.loadtxt(root + "yoochoose-clickssave1.txt", dtype=np.float)

        X = clicks[:1000000, 1:-1]
        y = clicks[:1000000, -1]

        alg = neighbors.KNeighborsClassifier(n_neighbors=1)
        alg.fit(X, y)

        print("Calculating test results...")
        sp = 9239729  # 6937297
        X2 = clicks[sp:, 1:-1]
        y2 = clicks[sp:, -1]
        user = clicks[sp:, 0]
        predict = alg.predict(X2)
        buy_user = []
        for i in range(len(user)):
            if predict[i] == 1:
                buy_user.append(user[i])


        # predict interesting items
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
            return clicklist, userlist, num_transaction


        # load transaction
        clicklist, userlist, num_transaction = load_transaction(
            root + "yoochoose-transaction-clicks.txt")
        # load click sequence
        clickseq, userlist, num_transaction = load_transaction(root + "yoochoose-sequence-clicks.txt")

        print("load data for markov chain")


        def initial_state(clickseq, clicklist):  # real sequence, unique sequence
            size = len(clicklist)
            state = np.zeros(size, dtype=np.float)
            for i in range(size):
                target = clicklist[i]
                num = clickseq.count(target)
                state[i] = num
            distribution = state / len(clickseq)
            return distribution


        def transform_probability(e, h):
            num = 0
            for transaction in clicklist:
                for i in range(len(transaction) - 1):
                    j = i + 1
                    if transaction[i] == e and transaction[j] == h:
                        num += 1
                        break
            p = num / num_transaction
            return p


        def sort_seprate(distribution, product):
            Z = zip(distribution, product)
            Z = sorted(Z, reverse=True)

            distribution1, product1 = zip(*Z)
            cut = int(len(product1) / 2) + 1
            return product1[:cut]
        # Markov chain
        file = open(root + "finalsolution.txt", "w")
        num_user = 0
        for user in buy_user:
            i = np.argwhere(np.array(userlist, dtype=np.float) == user)[0][0]
            num_user += 1
            print("====================================================")
            print("username: " + str(user) + " " + str(num_user) + "/" + str(len(buy_user)))
            buy_seq = clicklist[i]  # unique item order
            # Markov matrix(buy_seq>=5), else output all
            size = len(buy_seq)
            print("the number of items " + str(size))
            print("prediction: ")
            if size < 5:
                prediction = buy_seq
                print(prediction)
            else:
                graph = np.zeros((size, size), dtype=np.float)
                totaltime = size * size - size
                # count = 0
                for j in range(size):
                    for k in range(size):
                        if j != k:
                            item1 = buy_seq[j]
                            item2 = buy_seq[k]
                            p_i2_i1 = transform_probability(item1, item2)
                            graph[j][k] = p_i2_i1
                            # count+=1
                            # print(str(count) + "/" + str(totaltime))
                # calculate diaginal value
                for j in range(size):
                    line = graph[j]
                    total = np.sum(line)
                    graph[j][j] = 1 - total

                # current state matrix
                # get click sequences
                clickseq1 = clickseq[i]  # real sequence
                initial_distribution = initial_state(clickseq1, buy_seq)
                next_distribution = np.dot(initial_distribution, graph)
                prediction = sort_seprate(next_distribution, buy_seq)
                print(prediction)

            file.write(str(user) + " ")
            for l in range(len(prediction)):
                if l != len(prediction) - 1:
                    file.write(prediction[l] + " ")
                else:
                    file.write(prediction[l] + "\n")

        file.close()
    else:
        print("please input the correct command:")


