# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:45:45 2022

@author: alpha
"""

'''
pip install numpy scikit-image matplotlib tifffile
'''

import numpy as np
from ast import literal_eval
import json
from pprint import pprint as print
from zarr_stores.h5_nested_store import H5_Nested_Store

spool_file = r'H:\globus\pitt\bil\hillman\spool_examples\hicam\2023_04_24_HiCAMTest_sampleDataSet\hicam-scan-ROIb15_HiCAM FLUO_1875-ST-272.fli'
spool_file = r'/CBI_Hive/globus/pitt/bil/hillman/spool_examples/hicam/2023_04_24_HiCAMTest_sampleDataSet/hicam-scan-ROIb15_HiCAM FLUO_1875-ST-272.fli'

header_info = {
    # '{FLIMIMAGE}'
    # [INFO]
    'version': None,
    'compression': None,
    # '[LAYOUT]'
    'timeStamp': None,
    'CaptureVersion': None,
    'datatype': None,  # Should be UINT12
    'channels': None,
    'x': None,  # X axis pixel dimensions
    'y': None,  # Y axis pixel dimensions
    'z': None,
    'phases': None,
    'frequencies': None,
    'frameRate': None,
    'exposureTime': None,
    'deviceName': None,
    'deviceSerial': None,
    'deviceAlias': None,
    'packing': None,
    'hasDarkImage': None,
    'lutPath': None,
    'timestamps': None,  # Number of frames collected
    # '[DEVICE SETTINGS]'
    'Intensifier_PowerSwitch': None,
    'Intensifier_MCPvoltage': None,
    'Intensifier_MinimumMCPvoltage': None,
    'Intensifier_MaximumMCPvoltage': None,
    'Intensifier_SpecialCharacters': None,
    'Intensifier_AnodeCurrentLevelMicroAmps': None,
    'Intensifier_AnodeCurrentShutdownLevelMicroAmps': None,
    'Intensifier_AnodeCurrentProtectionSwitch': None,
    'Intensifier_UseCustomShutdownAnodeCurrentLevel': None,
    'Intensifier_CloseGateIfCameraIdleAvailable': None,
    'Intensifier_TECtargetTemperature': None,
    'Intensifier_TECheatsinkTemp': None,
    'Intensifier_TECobjectTemp': None,
    'Intensifier_GateOpenSwitch': None,
    'Intensifier_GateOpenTimeSeconds': None,
    'Intensifier_GateDelayTimeSeconds': None,
    'Intensifier_OutputAopenSwitch': None,
    'Intensifier_OutputAopenTimeSeconds': None,
    'Intensifier_OutputAdelayTimeSeconds': None,
    'Intensifier_OutputBopenSwitch': None,
    'Intensifier_OutputBopenTimeSeconds': None,
    'Intensifier_OutputBdelayTimeSeconds': None,
    'Intensifier_SyncMode': None,
    'Intensifier_GatingFixedFrequencyHz': None,
    'Intensifier_OutputAisPolarityPositive': None,
    'Intensifier_GateLoopModeSwitch': None,
    'Intensifier_BurstModeSwitch': None,
    'Intensifier_NumberOfBursts': None,
    'Intensifier_MultipleExposureSwitch': None,
    'Intensifier_NumberOfMultipleExposurePulses': None,
    'Intensifier_GateEnableInputSwitch': None,
    # 'headerLength': header_length,  # Index of last entry in the header
}
def read_header(file_name):
    '''
    Given a file name, read hicam header and extract parameters into a dictionary.  Make an effort to coerce
    values into appropriate types.

    Return dictionary

    Dictionary includes an extra key 'headerLength'.  All data after this index is image frame data.
    '''
    read_n = 0
    fileinfo = b''
    with open(file_name, 'rb') as f:
        while read_n < 40:
            a = f.read(1000)
            fileinfo += a
            if b'{END}' in fileinfo:
                header_length = fileinfo.index(b'{END}') + 5
                # location = str(fileinfo).index('{END}')
                print(f'Found the end of header at location {header_length}')

                # Convert fileinfo to a string and trim b' and remove anything after {END}
                header_string_end = str(fileinfo).index('{END}') + 5
                fileinfo = str(fileinfo)[2:header_string_end]
                break
            read_n += 1
        if read_n == 40:
            raise KeyError('Error while reading HICAM Header, Header Length too long?')

    raw_header_string = fileinfo
    # Extract header info
    fileinfo = fileinfo.split('\\n')
    for idx, ii in enumerate(fileinfo):
        print(ii)

    for ii in fileinfo:
        for key in header_info:
            test = key.lower() + ' = '
            if ii.lower().startswith(test):
                header_info[key] = ii[len(test)::]

    # Automatically convert values to appropriate types
    for key, value in header_info.items():
        try:
            header_info[key] = literal_eval(value)
        except Exception:
            pass
    header_info['headerLength'] = header_length
    print(header_info)
    return header_info, raw_header_string

def read_uint12(data_chunk, coerce_to_uint16_values=False):
    '''
    Since numpy does not understand uint12 data, this function takes a raw bytes object and reads the 12bit integer
    data into a uint16 array.

    Input:
        data_chunk: Byte string
        coerce_to_uint16_values (bool): if True outputs an array with values that are scaled to uint16
    Output:
         uint16 numpy array where each integer corresponds to the uint12 value (default)

    '''
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    if coerce_to_uint16_values:
        array *= 16
    return array


def read_data_file(file_name, header_info=None):

    if header_info is None:
        header_info, _ = read_header(file_name)

    ################################################################
    ## READ HICAM DATA IN ALL AT ONCE then convert to numpy z-stack
    ################################################################

    how_many_frames = header_info['timestamps']
    output = np.zeros((how_many_frames, header_info['y'], header_info['x']), 'uint16')

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame


    with open(spool_file, 'rb') as f:
        print(f'Reading {how_many_frames} frames')
        f.seek(header_info['headerLength'])
        multi_frame = f.read(how_many_frames * pixelInFrame_bit8)

    print(f'Forming Array')
    for idx in range(how_many_frames):
        where_to_start = idx * pixelInFrame_bit8
        data = multi_frame[where_to_start:where_to_start + pixelInFrame_bit8]

        # Data to uint16 where uint12 values have been scaled to uint16 values
        # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
        # canvas = read_uint12(data, coerce_to_uint16_values=True)
        canvas = read_uint12(data, coerce_to_uint16_values=False)

        output[idx] = canvas.reshape((header_info['y'], header_info['x']))

    return output

'''
Code below is designed to package a hicam spool file as a zarr
'''
def header_info_to_json(json_file,header_info):
    with open(json_file, 'w') as f:
        f.write(json.dumps(header_info, indent=4))

def send_hicam_to_zarr(hicam_file,zarr_location,compressor):

import os
from numcodecs import Blosc
import zarr

zarr_location = os.path.split(spool_file)[0] + '/out_zarr7' ## TEMP FOR TESTING

# Get Store
store = H5_Nested_Store(zarr_location)

# Dump header information to root of zarr store
hicam_file = spool_file
header_dict, raw_string = read_header(hicam_file)
header_json = json.dumps(header_dict, indent=4)
store['header.json'] = header_json.encode()
store['header_raw.txt'] = raw_string.encode()

image_data = read_data_file(hicam_file, header_info=None)

compressor = Blosc(
    cname='zstd',
    clevel=5,
    shuffle=1,
    blocksize=0
)

# chunks = (1,1,500,*image_data.shape[1:])
chunks = (500,*image_data.shape[1:])
chunks = (128,image_data.shape[1]//4, image_data.shape[2]//4)

# array = zarr.zeros(store=store, shape=(1,1,*image_data.shape), chunks=chunks, compressor=compressor, dtype=image_data.dtype)
array = zarr.zeros(store=store, shape=image_data.shape, chunks=chunks, compressor=compressor, dtype=image_data.dtype)
# array[0,0] = image_data
array[:] = image_data






# #######################################################################
# ## READ HICAM DATA 1 FRAME AT A TIME: adding each frame to numpy z-stack
# #######################################################################
#
# # Read 12bit data (numpy does not support this, so the following code deals with this by first interpreting 8bit)
# pixelInFrame_bit8 = int(header_info['x'] * header_info['y']/2 * 3) # Number of bits in frame
#
# scape_data = np.zeros((header_info['timestamps'], header_info['y'], header_info['x']), 'uint16')
# with open(spool_file, 'rb') as f:
#     for idx in range(header_info['timestamps']):
#         print(f'Reading {idx}')
#         where_to_start = header_info['headerLength'] + (idx * pixelInFrame_bit8)
#         f.seek(where_to_start)
#         data = f.read(pixelInFrame_bit8)
#
#         # Data to uint16 where uint12 values have been scaled to uint16 values
#         # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
#         canvas = read_uint12(data,coerce_to_uint16_values=True)
#
#         scape_data[idx] = canvas.reshape((header_info['y'], header_info['x']))



# ################################################################
# ## READ HICAM DATA IN BATCHES OF FRAMES: how_many_frames_at_once
# ################################################################
#
# scape_data = np.zeros((header_info['timestamps'], header_info['y'], header_info['x']), 'uint16')
# # Define the batch size for frames to be read off disk at once
# how_many_frames_at_once = 500
#
# import math
# how_many_groups_to_load = math.ceil(header_info['timestamps']/how_many_frames_at_once)
#
# for group in range(how_many_groups_to_load):
#
#     with open(spool_file, 'rb') as f:
#         print(f'Reading {group+1} of {how_many_groups_to_load}')
#         where_to_start = header_info['headerLength'] + (group * how_many_frames_at_once * pixelInFrame_bit8)
#         # if group > 0:
#         #     where_to_start += pixelInFrame_bit8
#         f.seek(where_to_start)
#         if group+1 == how_many_groups_to_load:
#             multi_frame = f.read()
#         else:
#             multi_frame = f.read(how_many_frames_at_once * pixelInFrame_bit8)
#
#     for local_idx in range(how_many_frames_at_once):
#         idx = local_idx + (group * how_many_frames_at_once)
#         print(f'Processing {idx}')
#         start = local_idx * pixelInFrame_bit8
#         stop = start+pixelInFrame_bit8
#         data = multi_frame[start:stop]
#
#         # Data to uint16 where uint12 values have been scaled to uint16 values
#         # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
#         canvas = read_uint12(data,coerce_to_uint16_values=True)
#
#         scape_data[idx] = canvas.reshape((header_info['y'], header_info['x']))
#
#         if idx+1 == scape_data.shape[0]:
#             break

