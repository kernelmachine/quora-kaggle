"""
create subsample (DUMMY DATA) for testing. should do really well on this one.
"""

import pandas as pd
import numpy as np 

df = pd.read_csv('data/train.csv')
df = df.dropna(axis=0)
x = df['question1'].sample(n=50000)
y = df['question2'].sample(n=50000)
question1 = np.concatenate([x.values, y.values], axis = 0)
question2 = np.concatenate([x.values, x.values], axis = 0)
is_duplicate = [1] * 50000 + [0] * 50000
df = pd.DataFrame({'is_duplicate':is_duplicate, 'question1':question1, 'question2':question2})
df.to_csv('data/subsample_test.csv', index = False)
