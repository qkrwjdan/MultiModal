from TextClassifier import TextClassifier
import torch
import pandas as pd
import numpy as np
import random

import os

SEED = 19

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
else: 
    device = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

master = pd.read_csv(os.path.join(os.getcwd(),"datasets","iemocap_text","session1_text_neu.csv"))
master_test = master[master["use"] == "test"]

print(master_test)

t = TextClassifier(session_nums = [1])
t.load_model()

for script_id in master_test["script_id"]:
   print(t.predict(script_id))