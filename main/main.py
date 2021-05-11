import pandas as pd
from libs.voter import Voter

import os
import time


master = pd.read_csv(os.path.join(os.getcwd(),'datasets','master','raw_session1.csv'))
master = master[['Clip_Name','Use','Label','Emotion']]

master_test = master[master['Use']=='test']

ans = 0
all = 0

print(master_test)

v = Voter()
acc_list = []
time_list = []

for i in range(10):
    start = time.time()
    ans = 0
    all = 0
    for j in range(len(master_test["Clip_Name"])):
        if j > i * 20 and j < (i+1) * 20:
            continue
        
        test_id = master_test["Clip_Name"].values[j]
        label = master_test["Label"].values[j]

        print(test_id)
        
        pred = v.voting(test_id)
        print("voting : ",pred)
        if pred == label:
            ans += 1
        all += 1
    
    acc = ans/all * 100
    acc_list.append(acc)
    print(i,") - ACC : ",ans / all * 100)
    time_list.append(time.time() - start)
    print("time : ",time.time() - start)
    break

print(acc_list)
print(time_list)

# start = time.time()

# for test_id,label in zip(master_test["Clip_Name"],master_test["Label"]):
#     print(test_id)

#     pred = v.voting(test_id)
#     print("voting : ",pred)
#     if pred == label:
#         ans += 1
#     all += 1

# print("ACC : ",ans / all * 100)
# print("time : ",time.time() - start)