import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

inputDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project3/comp-551-w2019-project-3-modified-mnist"
print(os.listdir(inputDirectory))

train_images = pd.read_pickle("/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project3/comp-551-w2019-project-3-modified-mnist/train_images.pkl")
train_labels = pd.read_csv("/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project3/comp-551-w2019-project-3-modified-mnist/train_labels.csv")
print(train_images.shape)

#Let's show image with id 16
img_idx = 13

plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
plt.imshow(train_images[img_idx])