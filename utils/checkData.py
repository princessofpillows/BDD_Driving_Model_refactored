
import os, glob
import numpy as np

from pathlib import Path
from builtins import any


def check_data(data_dir):
    '''
    Author: Jordan Patterson
    
    Function to check all relevant files in "data_dir" to ensure they are valid

    Parameters
    ----------

    data_dir : PurePath object
        Absolute path to the directory containing folders "videos", "info", "frame-10s" and "segmentation"

    '''
    
    # ensure we are searching a valid directory
    if not Path(data_dir).is_dir():
        print('Error: path to', data_dir, 'does not exist, change data_dir in config.py to a valid directory')
        return

    # get list of segmentation subdirectories
    subdirectories = os.listdir(data_dir / 'segmentation')
    # ensure subdirectories are valid
    if 'class_color' not in subdirectories or 'class_id' not in subdirectories or 'instance_color' not in subdirectories or 'instance_id' not in subdirectories or 'raw_images' not in subdirectories:
        print('Error: data directory', data_dir, 'does not contain all required folders (videos/info/frame-10s/segmentation)')
        return


    # get list of segmentation subdirectories
    subdirectories = os.listdir(data_dir / 'segmentation')
    # ensure subdirectories are valid
    if 'class_color' not in subdirectories or 'class_id' not in subdirectories or 'instance_color' not in subdirectories or 'instance_id' not in subdirectories or 'raw_images' not in subdirectories:
        print('Error: data directory', data_dir, 'segmentation does not contain all required folders (class_color, class_id, instance_color, instance_id, raw_images)')
        return

    # find paths for data
    videos = list(data_dir.glob('videos/*.mov'))
    info = list(data_dir.glob('info/*.json'))
    frames = list(data_dir.glob('frame-10s/*.jpg'))
    # find paths for labels
    class_colour = list(data_dir.glob('segmentation/class_color/*.png'))
    class_id = list(data_dir.glob('segmentation/class_id/*.png'))
    instance_colour = list(data_dir.glob('segmentation/instance_color/*.png'))
    instance_id = list(data_dir.glob('segmentation/instance_id/*.png'))
    raw_images = list(data_dir.glob('segmentation/raw_images/*.jpg'))
    # put paths in list
    data = [videos, info, frames, class_colour, class_id, instance_colour, instance_id, raw_images]

    # check that paths are not empty
    for path in data:
        if not path:
            print('Error: data directory ' + data_dir + ' does not contain data in required format in all folders')
            return

    # find largest dataset to parse
    names = max(data, key=len)
    index = data.index(names)

    # check that dataset is valid
    for name in range(len(names)):
        # for each video, get name without extension
        name = Path(data[index][name]).stem
        # check if video exists in all paths
        for d in data:
            # remove inconsistent videos
            if not any(name in index.stem for index in d):
                send_to_debug(data_dir, name)
                break
    
    # create array of random values, where length and range of s = length of datasets
    s = np.arange(np.asarray(videos).shape[0])
    np.random.shuffle(s)
    # shuffle data randomly
    videos = np.asarray(videos)[s]
    info = np.asarray(info)[s]
    frames = np.asarray(frames)[s]
    class_colour = np.asarray(class_colour)[s]
    class_id = np.asarray(class_id)[s]
    instance_colour = np.asarray(instance_colour)[s]
    instance_id = np.asarray(instance_id)[s]
    raw_images = np.asarray(raw_images)[s]

    return videos, info, frames, class_colour, class_id, instance_colour, instance_id, raw_images


def send_to_debug(data_dir, name):
    '''
    Author: Jordan Patterson
    
    Function to move all data with "name" to "debug" directory for debugging

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing folders "videos", "info", "frame-10s" and "segmentation"
    
    name : string
        Name of the files being moved

    '''
    print("Warning: Bad data for video " + name)
    try:
        # select new path for data
        newpath = data_dir / 'debug'
        # if new directory does not exist, create it
        if not newpath.exists():
            os.makedirs(newpath)
        # move data
        filepaths = [data_dir / 'videos' / (name + '.mov'), data_dir / 'info' / (name + '.json'), data_dir / 'frame-10s' / (name + '.jpg'), data_dir / 'segmentation' / 'class_color' / (name + '.png'), data_dir / 'segmentation' / 'class_id' / (name + '.png'), data_dir / 'segmentation' / 'instance_color' / (name + '.png'), data_dir / 'segmentation' / 'instance_id' / (name + '.png'), data_dir / 'segmentation' / 'raw_images' / (name + '.jpg')]
        for f in filepaths:
            os.rename(f, newpath / f.name)
        print('Warning: moving video ' + name + ' to', newpath)
    except FileNotFoundError:
        print('Warning: invalid path at ' + name)
    except FileExistsError:
        print("File already debugged: " + name)
        