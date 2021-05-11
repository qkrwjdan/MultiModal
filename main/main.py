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

start = time.time()

for test_id,label in zip(master_test["Clip_Name"],master_test["Label"]):
    print(test_id)

    pred = v.voting(test_id)
    print("voting : ",pred)
    if pred == label:
        ans += 1
    all += 1

print("ACC : ",ans / all * 100)
print("time : ",time.time() - start)