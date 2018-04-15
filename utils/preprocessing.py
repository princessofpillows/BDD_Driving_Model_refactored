
import glob, h5py, cv2, json
import numpy as np

from skimage.transform import resize
from tqdm import tqdm
from pathlib import Path, PurePath

from .checkData import check_data
from .processInfo import read_json


def package_data(data_dir):
    '''
    Function to search directories for folders containing video data to write in HD5 format

    The directory in any "data_dir" path must contain folders:
        - "videos": contains car dash cam videos in the .mov type
        - "info": contains driving data that accompanies "videos" in the .json type
        - "frame-10s": contains a photo which is taken at 10 seconds in its accompanying video in the .jpg type
        - "segmentation": contains the labels for the video data, in sub folders:
            -"class_color"
            -"class_id"
            -"instance_color"
            -"instance_id"
            -"raw_images"

    The names of the files in "videos", "info", "frame-10s" and "segmentation" must match each other at each index
    Any inconsistent files will be placed in a "debug" folder, which is ignored by the program

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing folders "videos", "info", "frame-10s" and "segmentation"

    '''

    # use pathlib
    data_dir = Path(data_dir)

    # checks all data in path specified at data_dir and returns the prepared data if valid
    videos, info, frames, class_colour, class_id, instance_colour, instance_id, raw_images = check_data(data_dir)

    # open file for r/w ('a' specifies not to overwrite)
    h5f = h5py.File('videoData.h5', 'a')

    # keep track of shortest video, and cut all videos to this length
    min_frames = int(cv2.VideoCapture(str[videos[0]]).get(cv2.CAP_PROP_FRAME_COUNT))

    # loops through all videos
    for i in tqdm(range(len(videos))):
        # gets name of video
        name = videos[i].stem
        # open video for frame processing
        video = cv2.VideoCapture(str(videos[i]))

        # ensure video opens successfully
        if not video.isOpened():
            video.release()
            # if it fails, move video to debug directory
            send_to_debug(data_dir, name)
            continue

        # get video metrics
        fps = int(np.rint(video.get(cv2.CAP_PROP_FPS)))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # update shortest video
        if total_frames < min_frames:
            min_frames = total_frames
        num_frames = int(np.rint(total_frames/fps))

        videodata = []
        count = 0
        # set refresh rate to 3hz
        hz = fps / 3
        
        # process video frame by frame
        while video.isOpened():
            # get frame of video
            ret, frame = cv2.VideoCapture.read(video)
            # check if we have reached end of video
            if ret != True or count == min_frames:
                break
            
            # record frame at 3hz with downsampled resolution
            if int(count % hz) == 0:
                frame = resize(frame, (640, 360, 3), preserve_range=True)
                videodata.append(frame)

            # count frames to ensure 3hz
            count += 1

        # close video object
        video.release()
        # get data ready to write
        video_data = np.asarray(videodata)
        info_data = read_json(info[i], num_frames, hz)
        if info_data is None:
            continue
        frame_data = cv2.imread(str(frames[i]), 1)
        class_colour_data = cv2.imread(str(class_colour[i]), 1)
        class_id_data = cv2.imread(str(class_id[i]), 1)
        instance_colour_data = cv2.imread(str(instance_colour[i]), 1)
        instance_id_data = cv2.imread(str(instance_id[i]), 1)
        raw_images_data = cv2.imread(str(raw_images[i]), 1)
        # resize images
        frame_data = resize(frame_data, (640, 360, 3))
        class_colour_data = resize(class_colour_data, (640, 360, 3))
        class_id_data = resize(class_id_data, (640, 360, 3))
        instance_colour_data = resize(instance_colour_data, (640, 360, 3))
        instance_id_data = resize(instance_id_data, (640, 360, 3))
        raw_images_data = resize(raw_images_data, (640, 360, 3))

        # write group for videoname
        try:
            group = h5f.create_group(name)
        # if group already exists, delete and recreate it
        except ValueError:
            print('Warning: group ' + name + ' already defined, resetting this group')
            del h5f[name]
            group = h5f.create_group(name)

        # write datasets to video group
        group.create_dataset('video', data=video_data, dtype='uint8')
        group.create_dataset('info', data=info_data)
        group.create_dataset('frame-10s', data=frame_data, dtype='uint8')
        group.create_dataset('class_colour', data=class_colour_data, dtype='uint8')
        group.create_dataset('class_id', data=class_id_data, dtype='uint8')
        group.create_dataset('instance_colour', data=instance_colour_data, dtype='uint8')
        group.create_dataset('instance_id', data=instance_id_data, dtype='uint8')
        group.create_dataset('raw_images', data=raw_images_data, dtype='uint8')

    # close file
    h5f.close()
