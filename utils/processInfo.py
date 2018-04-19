
import json, math
import numpy as np

from pathlib import Path, PurePath

from .checkData import send_to_debug


def read_json(filename, num_frames, hz):
    '''
    Author: Jordan Patterson

    Function to parse JSON info data and align with video

    Parameters
    ----------
    filename : PurePath object
        Absolute path to the json file being parsed

    num_frames : integer
        Number of frames in video

    hz : integer
        Refresh rate of video

    '''

    # parse locations from json file
    info = json.load(open(filename))
    locations = info['locations']
    
    # ensure json is valid
    if not check_info(info, locations):
        send_to_debug(Path(filename).parents[1], filename.stem)
        return

    data = {}
    # change data in locations to keys containing a 1d array of all associated values
    for key in locations[0].keys():
        data[key] = np.array([idx[key] for idx in locations]).ravel()

    data['startTime'] = info['startTime']
    data['endTime'] = info['endTime']

    velocity = np.zeros((len(data['course']), 2), dtype=np.float32)
    # create direction vector for every speed and course scalar
    for i in range(len(data['course'])):
        t = math.radians(data['course'][i])
        velocity[i, :] = np.array([math.sin(t) * data['speed'][i], math.cos(t) * data['speed'][i]])

    frame_velocity = np.zeros((num_frames, 2), dtype=np.float32)
    t_prev = 0
    
    # get velocity at current frame
    for frame in range(num_frames):
        # get current time in timestamp (location data is collected every second)
        t_cur = frame * 1000 / hz + data['startTime']

        # make sure current time is after first time
        if t_cur < data['timestamp'][0]:
            frame_velocity[frame, :] = velocity[0, :]
            continue

        # find timestamp before t_cur
        try:
            while data['timestamp'][t_prev + 1] < t_cur:
                t_prev += 1
        # last timestamp for locations reached
        except IndexError:
            frame_velocity[frame, :] = velocity[t_prev, :]
            continue

        # t_cur is between t_prev and t_next
        else:
            t_next = t_prev + 1
            # get difference between current time and previous/next time
            t1 = t_cur - data['timestamp'][t_prev]
            t2 = data['timestamp'][t_next] - t_cur
            # normalize differences in time
            r1 = t2 / (t1 + t2)
            r2 = t1 / (t1 + t2)
            # get current velocity between two timestamps
            frame_velocity[frame, :] = r1 * velocity[t_prev, :] + r2 * velocity[t_next, :]

    # align with framerate (3hz)
    frame_velocity = frame_velocity[::10, :]
    return frame_velocity


def check_info(info, locations):
    '''
    Author: Jordan Patterson
    
    Function to ensure json is valid

    Parameters
    ----------
    locations : array of dicts
        The locations and associated metrics of a video

    '''

    if len(locations) == 0:
        return False

    threshold = 2000
    # check if video starts too early or late
    if locations[0]['timestamp'] - info['startTime'] > threshold or info['endTime'] - locations[-1]['timestamp'] > threshold:
        return False

    failed = 0
    threshold = 1100
    prev_t = locations[0]['timestamp']
    # check if data valid
    for idx, loc in enumerate(locations):
        cur_t = int(loc['timestamp'])

        # fill missing values if possible
        for key in loc:
            if loc[key] == -1:
                failed += 1
                try:
                    if locations[idx + 1][key] != -1:
                        loc[key] = locations[idx + 1][key]
                    else:
                        loc[key] = locations[idx - 1][key]
                except IndexError:
                    loc[key] = locations[idx - 1][key]

        # ensure that timestamps are in reasonable range, the required # of keys exist, and enough speed/course data present to fill
        if cur_t - prev_t > threshold or cur_t - prev_t < 0 or len(loc.keys()) < 6 or failed == 3:
            return False

        prev_t = cur_t

    return True
