from libs.model import VideoClassifier
import pandas as pd
import numpy as np
import os
import time

master = pd.read_csv(os.path.join(os.getcwd(),'datasets','master','raw_session1.csv'))
master = master[['Clip_Name','Use','Label','Emotion']]

master_test = master[master['Use']=='test']

v = VideoClassifier(session_nums=[1],include_neu=False)
v.load_model('models/video/np_model_3class')

ans = 0
all = 0

start = time.time()

for test_id,label in zip(master_test["Clip_Name"],master_test["Label"]):
    print(test_id)

    video_predict = v.predict(test_id)
    print("video_predict : ",video_predict)
    if video_predict == label:
        ans += 1
    all += 1

print("ACC : ",ans / all * 100)
print("time : ",time.time() - start)