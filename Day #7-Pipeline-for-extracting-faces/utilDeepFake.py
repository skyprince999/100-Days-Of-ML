#pip install facenet-pytorch
#python3 -m pip install -U hachoir


import os
import re
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


def get_video_metadata(path):
    """
        Given a path, returns a dictionary of the video's metadata, as parsed by hachoir.
        Keys vary by exact filetype, but for an MP4 file on my machine,
        I get the following keys (inside of "Common" subdict):
            "Duration", "Image width", "Image height", "Creation date",
            "Last modification", "MIME type", "Endianness"

        Dict is nested - common keys are inside of a subdict "Common",
        which will always exist, but some keys *may* be inside of
        video/audio specific stream subdicts, named "Video Stream #1"
        or "Audio Stream #1", etc. Not all formats result in this
        separation.

        :param path: str path to video file
        :return: dict of video metadata
    """

    if not os.path.exists(path):
        raise ValueError("Provided path to video ({}) does not exist".format(path))

    parser = createParser(path)
    if not parser:
        raise RuntimeError("Unable to get metadata from video file")

    with parser:
        metadata = extractMetadata(parser)

        if not metadata:
            raise RuntimeError("Unable to get metadata from video file")
    
    metaInfo = metadata.exportPlaintext() 
    metadata_dict = dict()
    
    metaInfo = [md.replace("- ","").split(':') for md in metaInfo[1:]]
    
    for md in metaInfo:
        value = md[-1]
        key = '_'.join(md[:-1]).replace(" ", "")
        metadata_dict.update({key:value})
        
    #return metadata.exportPlaintext()
    return metadata_dict


import os
import time
import zipfile
from multiprocessing import Pool, cpu_count

# Function for extracting files from a compressed zip file 
def extractZip(zipFilePath, extractPath = '/home/jupyter/'):
    """
        Given a zipped File, extracts all the contents of the compressed file to the extractPath.
        It's a wrapper on zipfile package
        This has been only tested for zip file formats
        Other file formats have not been tested. But may work if supported by zipfile package
        
        :param zipFilePath: str path (full Path) to zipped file
        :param extractPath: str path (full Path) to output, default is /home/jupyter/ 
        :return: List of files if successful or False if exception was raised
    """
    
    print('Extracting File: {} to Path:  {}'.format(zipFilePath, extractPath) )
    
    if not os.path.isdir(extractPath):
        os.mkdir(extractPath)
    try:
        start =time.time()
        zip_ref = zipfile.ZipFile(zipFilePath, 'r')
        
        fileList = zip_ref.namelist()
        zip_ref.extractall(path=extractPath)
        zip_ref.close()
        end =time.time()
        print("Elapsed Time:", end- start)
    except:
        return False
    return fileList


import os
import cv2
import time
import numpy as np
from tqdm.notebook import tqdm

def extractFrames( video, frameSample = 10):
    """
        Given a zipped File, extracts all the contents of the compressed file to the extractPath.
        It's a wrapper on zipfile package
        This has been only tested for zip file formats
        Other file formats have not been tested. But may work if supported by zipfile package
        
        :param video: str path to video file
        :param frameSample: sampling rate for the video file. Default is 10, so 10 equi-distant frames will be extracted from the video
        :return: stacked N samples as given by the param frameSample
    """
    nFrames = list()
    
    start = time.time()
    reader = cv2.VideoCapture(video)
    noOfFrames = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frameIndex = np.round(np.linspace(0, noOfFrames, frameSample, endpoint=False)).astype(int)
    
    for idx in frameIndex:
        reader.set(cv2.CAP_PROP_POS_MSEC, idx)
        success,image = reader.read()
        
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nFrames.append(image)
            
    reader.release()    
    nFrames = np.stack(nFrames)
    end = time.time()
    #print('Elapsed Time: ', (end-start))
    return nFrames

from PIL import Image
import torch
from facenet_pytorch import MTCNN

def detect_facenet_pytorch(images, batch_size):
    """
        Given a zipped File, extracts all the contents of the compressed file to the extractPath.
        It's a wrapper on zipfile package
        This has been only tested for zip file formats
        Other file formats have not been tested. But may work if supported by zipfile package
        
        :param detector: detector used for detecting faces 
        :param images: numpy stack of images 
        :param batch_size: batchd detection 
        :return: tensor of faces detected
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = MTCNN(device=device, post_process=False)
    
    start =time.time()
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        faces.extend(detector(imgs_pil))
    end =time.time()
    #print("Elapsed Time: ", (end-start))
    images = torch.stack(faces).permute(0, 2, 3, 1).int().numpy()
    
    return images

import os
import cv2
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt 

def saveFaces(images, videoFile, faceDir = '/home/jupyter/faces/'):
    """
        Given a zipped File, extracts all the contents of the compressed file to the extractPath.
        It's a wrapper on zipfile package
        This has been only tested for zip file formats
        Other file formats have not been tested. But may work if supported by zipfile package
        
        :param images: list of cropped faces 
        :param videoFile: name of original video file (don't pass the full path just the filename)
        :param faceDir: directory to store faces, /home/jupyter/faces/ is default  
        :return: True if successful operation
    """
    if not os.path.isdir(faceDir):
        os.mkdir(faceDir)
    
    fileList = list()    
    baseFilename = faceDir + videoFile.split('.')[0] + '_frame' 
    for idx in range(len(images)):
        filename = baseFilename + str(idx) + '.jpg'
        img = np.array(images[idx], dtype=np.uint8)
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img ) 
        fileList.append(filename)
    return fileList