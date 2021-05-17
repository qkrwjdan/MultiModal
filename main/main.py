import pandas as pd
from libs.voter import Voter

import os
import time
import csv

v = Voter()
acc_list = []
time_list = []

test_dataset_path = os.path.join(os.getcwd(),"datasets","master","testDataset")

dataset_list = os.listdir(test_dataset_path)

dataset_list.sort()

acc_list = []
time_list = []

for path in dataset_list:
    test_dataset = os.path.join(test_dataset_path, path)
    df = pd.read_csv(test_dataset)
    print("="*10,test_dataset,"="*10)

    ans = 0
    all = 0

    start = time.time()

    for test_id, label in zip(df["Clip_Name"],df["Label"]):
        print("="*20)
        print("Data ID : ",test_id)
        print("Actual : ",label)

        pred = v.voting(test_id)
        print("Multimodal Output : ",pred)
        print("="*20)


        if label == pred:
            ans += 1
        all += 1
    
    acc = ans / all * 100
    print("Total Accuracy : ",acc)
    acc_list.append(acc)
    
    time_list.append(time.time() - start)
    print("time : ",time.time() - start)

print("="*30)
print("Total Accuracy = ",sum(acc_list) / len(acc_list))
print("="*30)

with open("result.csv","w") as f:
    wr = csv.writer(f)
    wr.writerow(["ID","ACC","TIME"])

    for i in range(len(acc_list)):
        wr.writerow([i+1,acc_list[i],time_list[i]])
