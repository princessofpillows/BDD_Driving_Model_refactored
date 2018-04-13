
import os, glob
import numpy as np

from builtins import any

def check_data(data_dir):
    '''
    Function to check all relevant files in "data_dir" to ensure they are valid

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing folders "videos", "info", "frame-10s" and "segmentation"

    '''
    # ensure we are searching a valid directory
    if not os.path.exists(data_dir):
        print('Error: path to ' + data_dir + ' does not exist, change data_dir in config.py to a valid directory')
        return

    # get list of subdirectories
    subdirectories = os.listdir(data_dir)
    # ensure subdirectories are valid
    if 'videos' not in subdirectories or 'info' not in subdirectories or 'frame-10s' not in subdirectories or 'segmentation' not in subdirectories:
        print('Error: data directory ' + data_dir + ' does not contain all required folders (videos/info/frame-10s/segmentation)')
        return

    # get list of segmentation subdirectories
    subdirectories = os.listdir(data_dir + 'segmentation')
    # ensure subdirectories are valid
    if 'class_color' not in subdirectories or 'class_id' not in subdirectories or 'instance_color' not in subdirectories or 'instance_id' not in subdirectories or 'raw_images' not in subdirectories:
        print('Error: data directory ' + data_dir +  'segmentation does not contain all required folders (class_color, class_id, instance_color, instance_id, raw_images)')
        return

    # find paths for data
    videopaths = glob.glob(data_dir + 'videos/*.mov')
    infopaths = glob.glob(data_dir + 'info/*.json')
    framepaths = glob.glob(data_dir + 'frame-10s/*.jpg')
    # find paths for labels
    class_colour = glob.glob(data_dir + 'segmentation' + '/class_color/*.png')
    class_id = glob.glob(data_dir + 'segmentation' + '/class_id/*.png')
    instance_colour = glob.glob(data_dir + 'segmentation' + '/instance_color/*.png')
    instance_id = glob.glob(data_dir + 'segmentation' + '/instance_id/*.png')
    raw_images = glob.glob(data_dir + 'segmentation' + '/raw_images/*.jpg')
    # put paths in list
    data = [videopaths, infopaths, framepaths, class_colour, class_id, instance_colour, instance_id, raw_images]

    # check that paths are not empty
    for path in data:
        if not path:
            print('Error: data directory ' + data_dir + ' does not contain data in required format in all folders')
            return

    # find largest dataset to parse
    videos = max(data, key=len)
    index = data.index(videos)
    # check that dataset is valid
    x = 0
    for video in range(len(videos)):
        # for each video
        name = os.path.basename(data[index][video])[:-4]
        # check if video exists in all paths
        for d in range(len(data)):
            # remove inconsistent videos
            if not any(name in index for index in data[d]):
                send_to_debug(data_dir, name)
                break
    
    # create array of random values, where length and range of s = length of datasets
    s = np.arange(np.asarray(videopaths).shape[0])
    np.random.shuffle(s)
    # shuffle data randomly
    for d in range(len(data)):
        data[d] = np.asarray(data[d])[s]

    # return the datasets
    return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]


def send_to_debug(data_dir, name):
    '''
    Function to move all data with "name" to "debug" directory for debugging

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing folders "videos", "info", "frame-10s" and "segmentation"
    
    name : string
        Name of the files being moved

    '''
    try:
        # select new path for data
        newpath = data_dir + '/debug/'
        # if new directory does not exist, create it
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        # move data
        filepaths = [data_dir + 'videos/' + name + '.mov', data_dir + 'info/' + name + '.json', data_dir + 'frame-10s/' + name + '.jpg', data_dir + 'segmentation' + '/class_color/' + name + '.png', data_dir + 'segmentation' + '/class_id/' + name + '.png', data_dir + 'segmentation' + '/instance_color/' + name + '.png', data_dir + 'segmentation' + '/instance_id/' + name + '.png', data_dir + 'segmentation' + '/raw_images/' + name + '.jpg']
        for f in filepaths:
            os.rename(f, newpath + os.path.basename(f))
        print('Warning: moving video ' + name + ' to ' + newpath)
    except FileNotFoundError:
        print('Warning: invalid path at ' + filepath)
