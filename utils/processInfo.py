
import json

def read_json(filename):
    '''
    Function to parse JSON info data and align with video

    Parameters
    ----------
    filename : string
        Absolute path to the json file being parsed

    '''

    info = json.load(open(filename))
    return json.dumps(info)
