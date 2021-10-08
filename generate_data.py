# %%
import requests
import json
import pandas as pd
import numpy as np
import os
import urllib.request
from cv2 import imread
from tqdm.notebook import  tqdm
import warnings
warnings.filterwarnings("ignore")

base_dir = ''
os.mkdir(os.path.join(base_dir, 'punks'))

# %%
def data(offset):
  # opensea has a limit of 20 punks per api request
  url = "https://api.opensea.io/api/v1/assets"
  querystring = {"order_direction":"desc","offset":f"{offset}","limit":"20","collection":"cryptopunks"}
  response = requests.request("GET", url, params=querystring)
  d = json.loads(response.text)
  
  metadata = pd.DataFrame()
  try:  
    # loop through each result
    for i in range(0,20):
      df = pd.json_normalize(d['assets'][i], sep='_')
      name = df['name'].iloc[0]
      traits = pd.json_normalize(df['traits'].iloc[0], sep='_')
      traits = traits.sort_values(by='trait_type')
      for j,i in enumerate(traits['trait_type']):
        if i == 'accessory':
          traits['trait_type'].iloc[j] = f'accessory{j}'
      traits = pd.DataFrame(traits[['trait_type','value']].T.values[1:], columns = traits[['trait_type','value']].T.values[0])
      traits['name'] = name
      metadata = pd.concat([metadata,traits])
      
      # save image to punks directory
      if not os.path.exists(os.path.join(base_dir,"punks/{}.png".format(name))):
          urllib.request.urlretrieve( df['image_url'].iloc[0], f"punks/{name}.png")
  except:
      pass
  return metadata

print('Downloading CryptoPunks...')
metadata = pd.DataFrame()
for i in tqdm(range(0,10020,20)):
  #print('Punks downloaded: {}'.format(len(os.listdir(os.path.join(base_dir,"punks")))))
  metadata = pd.concat([metadata,data(i)])

# %%

for i in metadata['type'].unique():
  metadata[i] = np.where(metadata['type']==i,1,0)
metadata

for j in range(5):
  for i in metadata[f'accessory{j}'].unique():
    try:
      print(metadata[i].iloc[0])
      metadata[i] = np.where(metadata[f'accessory{j}']==i,1,metadata[i])
    except:
      metadata[i] = np.where(metadata[f'accessory{j}']==i,1,0)

metadata = metadata.drop(columns=[np.nan])

cols = ['name']  + list(metadata.columns[-92:])
metadata[cols].to_csv(os.path.join(base_dir, 'punks_dummies.csv'),index=False)
print('Saved metadata file.')
# %%

df = pd.read_csv(os.path.join(base_dir,'punks_dummies.csv'))
df['id'] = df['name'].apply(lambda x:int(str(x).split('#')[-1]))
df = df.sort_values(by='id')
# %%

from skimage.transform import resize

X3 = []
Y3 = []
X2 = []
Y2 = []

print('Concatenating Images.')
for i in tqdm(range(10001)):
    row = df.iloc[i]
    name = row['name']

    X = imread(os.path.join(base_dir,f'raw/punks/{name}.png'))
    X = resize(X, (42, 42))
    Y = np.array(row.iloc[1:-1])
    X = X.reshape((1,X.shape[0],X.shape[1],X.shape[2]))
    
    if X2 == []:
        X2 = X.astype('float16')
        Y2 = Y
    else:
        X2 = np.concatenate((X2,X)).astype('float16')
        Y2 = np.vstack((Y2,Y))
    
    if i>9900:
        X3 = np.concatenate((X3,X)).astype('float16')
        Y3 = np.vstack((Y3,Y))
    else:
        if i%100==0:
            if X3 == []:
                X3 = X2
                Y3 = Y2
            else:
                X3 = np.concatenate((X3,X)).astype('float16')
                Y3 = np.vstack((Y3,Y))
            X2 = []
            Y2 = []
# %%
np.save('x_punk', X3)
np.save('y_punk', Y3)

print('Saved Arrays.')
print('generate_data.py completed')
# %%
