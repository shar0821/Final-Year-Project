from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
from PIL import Image
import math

csv_file=pd.read_csv('./results_VGG19.csv', names=['image_name', 'Glaucoma', 'Not Glaucoma'])

csv_file=csv_file.drop(index=0)

y = []
yhat = []

for i, row in csv_file.iterrows():
    img_name = row['image_name']
    gl = row['Glaucoma']
    not_gl = row['Not Glaucoma']
    
    if("_g_" in img_name):
        y.append(1)
        yhat.append(round(float(gl)))
        
    else:
        yhat.append(round(float(not_gl)))
        y.append(0)


y = np.array(y)
yhat = np.array(yhat)

print("y", y)
print("yhat", yhat)

acc = accuracy_score(y, yhat)
print("Accuracy: {:.2f}%".format(acc * 100))

f1=f1_score(y, yhat)
print("F1: {:.2f}%".format(f1 * 100))
