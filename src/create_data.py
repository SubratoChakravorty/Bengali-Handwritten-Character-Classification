import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as aug
from PIL import Image
from sklearn.model_selection import train_test_split


# extracts region of interest and resize to 64 * 64
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size=64
    angle=0
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):

            image=df.loc[df.index[i]].values.reshape(137,236)
            # Centering
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # Scaling
            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)
            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # Brightness
            augBright = aug.RandomBrightnessContrast(p=1.0)
            image = augBright(image=image)['image']
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized


# for i in range(4):
#     df = pd.read_parquet(f'../input/train_image_data_{i}.parquet', engine='pyarrow')
#     image_ids = df.image_id.values
#     df = df.drop('image_id', axis=1)
#     resize_df = resize(df)
#
#     image_values = resize_df.values
#     for j, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
#         img = Image.fromarray(image_values[j,:].reshape(64, 64))
#
#         img.save(f'../input/images/{image_id}.png')


# Split train.csv into train, test and val set
df = pd.read_csv('../input/train.csv')
image_ids = df.image_id
df = df.drop('grapheme', axis=1)
train_ids, test_ids = train_test_split(image_ids, test_size=0.2)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1)
train_df = df[df['image_id'].isin(train_ids.values)]
val_df = df[df.image_id.isin(val_ids.values)]
test_df = df[df.image_id.isin(test_ids.values)]

train_df['filepath'] = train_df['image_id'] + '.png'
val_df['filepath'] = val_df['image_id'] + '.png'
test_df['filepath'] = test_df['image_id'] + '.png'

train_df.to_csv('../input/train_df.csv')
val_df.to_csv('../input/val_df.csv')
test_df.to_csv('../input/test_df.csv')
