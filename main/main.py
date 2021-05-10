import pandas as pd
from libs.voter import Voter

import os


master = pd.read_csv(os.path.join(os.getcwd(),"datasets","master","raw_session1.csv"))

master_test = master[master["Use"] == "test"]

print(master_test)

