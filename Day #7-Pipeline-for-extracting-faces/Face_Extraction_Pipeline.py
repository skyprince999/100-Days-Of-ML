#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import json
import pandas as pd
import shutil
from tqdm.notebook import tqdm


# In[2]:


import sys

sys.path.append('/home/jupyter/')
import utilDeepFake


# In[3]:


zipFiles = os.listdir('/mnt/disks/backup')
zipFiles = [f for f in zipFiles if f.split('.')[-1] == 'zip']
len(zipFiles)


# In[6]:


shutil.rmtree('dfdc_train_part_5')


# In[7]:


for idx in tqdm(range(4,50)):
    zipFilePath = '/mnt/disks/backup/' + zipFiles[idx]
    zip_ref = utilDeepFake.extractZip(zipFilePath)
    
    extractFolder = zip_ref[0].split('/')[0]
    zipList = os.listdir(extractFolder)
    print(len(zipList))
    
    metaFile = extractFolder +"//metadata.json"
    with open(metaFile) as fp:
        data = json.load(fp)
        metaDF = pd.DataFrame(data)
        metaDF = metaDF.T
    
    count = 0
    faceFile = list()
    flag = list()
    origFile = list()    
    
    videoFiles = metaDF.index

    for vfile in tqdm(videoFiles):

        if 'json' not in vfile:
            label = metaDF.loc[vfile, :'label'].values[0]

            vfile = extractFolder + '//'+ vfile
            print("Processing: ",vfile)

            try:
                frames = utilDeepFake.extractFrames(vfile, 10)
                faces = utilDeepFake.detect_facenet_pytorch(frames, 16)
                fileList = utilDeepFake.saveFaces(faces, vfile.split('/')[-1])
            except:
                continue
            faceFile.extend(fileList)
            flag.extend(len(fileList)*[label])
            origFile.extend([vfile]*len(fileList))
    
    print(len(faceFile))
    print(len(flag))

    faceDF = pd.DataFrame({'faceFile': faceFile, 'label': flag})
    
    processedFile = faceDF['faceFile'].values
    faceDF['processedFile'] = [f.split('/')[-1].split('_')[0] + '.mp4' for f in processedFile]

    excelFile = 'faceLabels_part_' + extractFolder.split('_')[-1] + '.xlsx'
    
    faceDF.to_excel(excelFile)
    
    shutil.rmtree(extractFolder)


# In[8]:


idx


# In[10]:


zipFiles[22]


# In[ ]:




