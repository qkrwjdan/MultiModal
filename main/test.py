import pandas as pd

import os
import time
import csv
import random


master = pd.read_csv(os.path.join(os.getcwd(),'datasets','master','raw_session1.csv'))
master = master[['Clip_Name','Use','Label','Emotion']]

master_test = master[master['Use']=='test']

tmpList = []

print(master_test)

for i in range(10):
    
    tmpList = []

    with open("datasets/master/testDataset/testData{i}.csv".format(i=i+1),"w") as f:
        wr = csv.writer(f)
        wr.writerow(["Clip_Name","Label","Use"])

        for j in range(len(master_test["Clip_Name"])):
            if j > i * 20 and j < (i+1) * 20:
                continue
            

            test_id = master_test["Clip_Name"].values[j]
            label = master_test["Label"].values[j]
            use = master_test["Use"].values[j]

            tmpList.append([test_id,label,use])
        
        random.shuffle(tmpList)

        for j in range(len(tmpList)):
            wr.writerow(tmpList[j])



