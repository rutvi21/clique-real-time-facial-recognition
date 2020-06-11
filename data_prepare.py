import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import os

# making folders

inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', 'train', inner_name), exist_ok=True)

# to keep count of each category
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

def atoi(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n

for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()

    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    if df['emotion'][i] == 0:
        img.save('train/angry/im' + str(angry) + '.png')
        angry += 1
    elif df['emotion'][i] == 1:
        img.save('train/disgusted/im' + str(disgusted) + '.png')
        disgusted += 1
    elif df['emotion'][i] == 2:
        img.save('train/fearful/im' + str(fearful) + '.png')
        fearful += 1
    elif df['emotion'][i] == 3:
        img.save('train/happy/im' + str(happy) + '.png')
        happy += 1
    elif df['emotion'][i] == 4:
        img.save('train/sad/im' + str(sad) + '.png')
        sad += 1
    elif df['emotion'][i] == 5:
        img.save('train/surprised/im' + str(surprised) + '.png')
        surprised += 1
    elif df['emotion'][i] == 6:
        img.save('train/neutral/im' + str(neutral) + '.png')
        neutral += 1
