# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:45:45 2022

@author: alpha
"""

'''
This module provides tools to read hicam data files and convert the data to compressed zarr arrays 

hicam data is saved as UINT12.  A function converts UINT12>>UINT16 then appropriately rescales the data to
UINT16 values. 0:4095 >> 0:65534.  Data are stored as UINT16. To recover the original UINT12 values simply divide by 16

Arrays are saved according to hicam format: thus axes x,y,and z are representative of the cameras coordinates.  
X and Y are lateral axes (each 2D frame capture), Z represents a stack of many frames.

SCAPE data acquisition often requires that these axes are rearranged to represent acquisition axes.  For example a
mapping of hicam axes to SCAPE acquisition axes would be:
{
# hicam:scape #
'y':'z',
'x':'y',
'z':'x'
}
'''

import numpy as np
from ast import literal_eval
import json
from pprint import pprint as print
import os
from numcodecs import Blosc
# from numcodecs import blosc
# blosc.set_nthreads(16)
from zarr_stores.h5_nested_store import H5_Nested_Store
import zarr
from skimage import io

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

def read_uint12(data_chunk, coerce_to_uint16_values=True):
    '''
    Since numpy does not understand uint12 data, this function takes a raw bytes object and reads the 12bit integer
    data into a uint16 array.

    Input:
        data_chunk: Byte string
        coerce_to_uint16_values (bool): if True outputs an array with values that are scaled to uint16

        In general this should remain True.  Thus, the image is representative of a conversion to UINT16 precision
        and any conversion to other precision images (ie float for processing) will appropriately represent the
        original data. *Manual conversion to the original uint12 values can be obtained by division by 16

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

    pixelInFrame_bit8 = int(header_info['x'] * header_info['y'] / 2 * 3)  # Number of bits in frame

    how_many_frames = header_info['timestamps']
    if how_many_frames is None:
        with open(spool_file, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size_of_file = f.tell()
            print(f'{size_of_file=}')
            header_len = header_info['headerLength']
            print(f'{header_len=}')
            data_size = size_of_file - header_len
            print(f'{data_size=}')
            num_frames_remainder = data_size%pixelInFrame_bit8
            print(f'{num_frames_remainder=}')
            how_many_frames = data_size//pixelInFrame_bit8
            print(f'{how_many_frames=}')
    output = np.zeros((how_many_frames, header_info['y'], header_info['x']), 'uint16')


    with open(spool_file, 'rb') as f:
        size_of_file = f.tell()
        print(f'{size_of_file=}')
        print(f'Reading {how_many_frames} frames')
        f.seek(header_info['headerLength'])
        multi_frame = f.read(how_many_frames * pixelInFrame_bit8)

    print(f'Forming Array')
    for idx in range(how_many_frames):
        where_to_start = idx * pixelInFrame_bit8
        data = multi_frame[where_to_start:where_to_start + pixelInFrame_bit8]

        # Data to uint16 where uint12 values have been scaled to uint16 values
        # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
        canvas = read_uint12(data, coerce_to_uint16_values=True)
        # canvas = read_uint12(data, coerce_to_uint16_values=False)

        output[idx] = canvas.reshape((header_info['y'], header_info['x']))

    return output

'''
Code below is designed to package a hicam spool file as a zarr
'''
def header_info_to_json(json_file,header_info):
    with open(json_file, 'w') as f:
        f.write(json.dumps(header_info, indent=4))

def send_hicam_to_zarr(hicam_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1):

    compressor = Blosc(
        cname=compressor_type,
        clevel=compressor_level,
        shuffle=shuffle,
        blocksize=0
    )

    # zarr_location = os.path.split(spool_file)[0] + '/out_zarr7' ## TEMP FOR TESTING

    # Get Store
    store = H5_Nested_Store(zarr_location)

    # Dump header information to root of zarr store
    header_dict, raw_string = read_header(hicam_file)
    header_json = json.dumps(header_dict, indent=4)
    store['header.json'] = header_json.encode()
    store['header_raw.txt'] = raw_string.encode()

    image_data = read_data_file(hicam_file, header_info=None)

    # Define chunks: Groups of 128 frames, then each color channel is divided into 4 chunks (2x2)
    chunks = (128,image_data.shape[1]//4, image_data.shape[2]//4)

    # array = zarr.zeros(store=store, shape=(1,1,*image_data.shape), chunks=chunks, compressor=compressor, dtype=image_data.dtype)
    array = zarr.zeros(store=store, shape=image_data.shape, chunks=chunks, compressor=compressor, dtype=image_data.dtype)
    # array[0,0] = image_data
    array[:] = image_data


def get_hicam_zarr(zarr_location):
    store = H5_Nested_Store(zarr_location, 'r')
    array = zarr.open(store,'r')
    print(array)
    return array

class open_hicam_array_as_aligned_color_dataset:

    '''
    Present the hicam data (z,y,x) as a multicolor 3D array.  The class handles all the trimming and alignment of
    channels in (y,x)

    Input
    -----> X (hicam axis)
    --------------------- '
    '    c0   '    c1   ' '
    '         '         ' V
    ---------------------
    '    c2   '    c3   ' Y (hicam axis)
    '         '         '
    ---------------------

    Output
    -----------
    ' c0,1,2,3'
    '         '
    -----------

    '''

    def __init__(self,zarr_location):

        self.store = H5_Nested_Store(zarr_location, 'r')
        self.array = zarr.open(self.store, 'r')

        # Output array details
        self.shape = (4,self.array.shape[0],self.array.shape[1]//2,self.array.shape[2]//2)
        self.dtype = self.array.dtype
        self.size = self.array.size
        self.ndims = 4
        self.chunks = (4,self.array[0],self.shape[2],self.shape[3])

    def __getitem__(self, item):
        print(f'In slice: {item}')

        if isinstance(item, int):
            item = (slice(item,item+1), slice(0, self.shape[1]), slice(0, self.shape[2]), slice(0,self.shape[3]))

        elif isinstance(item, slice):
            item = (item, slice(0, self.shape[1]), slice(0, self.shape[2]), slice(0,self.shape[3]))

        elif isinstance(item, tuple):

            tmp_slice = []

            for idx,ii in enumerate(item):
                if isinstance(ii, int):
                    tmp_slice.append(slice(ii, ii+1))
                elif isinstance(ii,slice):
                    tmp_slice.append(ii)

            idx += 1
            while idx < self.ndims:
                tmp_slice.append(slice(0, self.shape[idx]))
                idx += 1
                # print(idx)

            item = tmp_slice

        # print(f'Out slice: {item}')
        ## ITEM is now the corrected slice for the virtual array


        ## Actually get data from hicam file and shape into the new array shape
        canvas = np.zeros(
            shape=(
            item[0].stop - item[0].start,
            item[1].stop - item[1].start,
            item[2].stop - item[2].start,
            item[3].stop - item[3].start
        ),
            dtype=self.array.dtype
        )
        # print(canvas.shape)

        for clr_idx in range(canvas.shape[0]):
            clr = clr_idx + item[0].start
            # print(f'CLR index = {clr}')
            if clr == 0:
                color_slice = (
                    slice(0,self.shape[2]),
                    slice(0,self.shape[3])
                )
            if clr == 1:
                color_slice = (
                    slice(0, self.shape[2]),
                    slice(self.shape[3], self.shape[3]*2)
                )
            if clr == 2:
                color_slice = (
                    slice(self.shape[2], self.shape[2]*2),
                    slice(0, self.shape[3])
                )
            if clr == 3:
                color_slice = (
                    slice(self.shape[2], self.shape[2]*2),
                    slice(self.shape[3], self.shape[3]*2)
                )

            out = self.array[item[1],color_slice[0],color_slice[1]]
            # print(out.shape)
            # print(out)
            out = out[:,item[-2],item[-1]]
            # print(out.shape)
            canvas[clr_idx] = out
            # print(canvas.shape)

        return canvas.squeeze()



####  Image alignments methods
# pip install numpy scikit-image matplotlib SimpleITK

import SimpleITK as sitk
from skimage import io, img_as_float32, img_as_float, img_as_uint
import numpy as np
import time
def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def sitk_align_translation(fixed, moving, output_offsets=False):
    '''
    Input:
        fixed: numpy array (same shape as moving)
        moving: numpy array (same shape as fixed)

    Output:
        If output_offsets == False (default), an aligned image is returned
        If output_offsets == True, a tuple of pixel offsets is returned

        Images are returned in the same dtype as input
    '''

    dtype = moving.dtype
    # Convert numpy arrays to sitk images
    if fixed.dtype != float:
        fixed = img_as_float32(fixed)
    if moving.dtype != float:
        moving = img_as_float32(moving)
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)
    fixed.SetOrigin((0, 0))
    moving.SetOrigin((0, 0))

    # Calculate alignment
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    # R.SetMetricAsMattesMutualInformation(50)
    # R.SetMetricAsJointHistogramMutualInformation()
    # R.SetMetricAsANTSNeighborhoodCorrelation(2)
    # R.SetMetricAsMeanSquares()
    # R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    # R.SetInterpolator(sitk.sitkNearestNeighbor)
    R.SetInterpolator(sitk.sitkLinear)

    # Pyramidal registration
    R.SetShrinkFactorsPerLevel([6, 2, 1])
    R.SetSmoothingSigmasPerLevel([6, 2, 1])

    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)
    # return outTx
    if output_offsets:
        # Return revered tuple of pixel offsets (sitk and numpy axes are reversed)
        # Returned axes are in order (y,x)
        offsets = outTx.GetParameters()[::-1]
        # print(offsets)
        return offsets

    # Produce aligned image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    R.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)

    # Return numpy array of aligned image
    out = sitk.GetArrayFromImage(out)
    if dtype == out.dtype:
        return out
    if dtype == np.dtype('uint16'):
        return img_as_uint(out)
    if dtype == np.dtype('float32'):
        return img_as_float32(out)
    if dtype == float:
        return img_as_float(out)


def calculate_channels_shift(multi_channel_z_stack, reference_channel=None):
    from skimage.registration import phase_cross_correlation

    if reference_channel is None:
        reference_channel = 0

    shift_dict = {
        0:[],
        1: [],
        2: [],
        3: []
    }
    for ch_idx in range(multi_channel_z_stack.shape[0]):

        ref_array = multi_channel_z_stack[reference_channel]
        reference = ref_array[0]
        if ch_idx != reference_channel:

            moving_array = multi_channel_z_stack[ch_idx]
            moving = moving_array[0]
            for z_idx in range(moving_array.shape[0]):
                # print(z_idx)
                if z_idx%10 == 0:
                    # print(f'Getting Reference and Moving images')
                    reference[:] = ref_array[z_idx]
                    moving[:] = moving_array[z_idx]
                    # print(f'Aligning Image {z_idx} of {multi_channel_z_stack.shape[1]}')
                    # shift, error, phasediff = phase_cross_correlation(reference,moving)
                    # print(f'Shift = {shift}, error = {error}, phasediff = {phasediff}')
                    shift = sitk_align_translation(reference, moving, output_offsets=True)
                    print(f'{z_idx} of {moving_array.shape[0]}; Shift = {shift}')
                    shift_dict[ch_idx].append(shift)

    return shift_dict


def calculate_channel_shifts(zarr_spool_location, anchor_channel):

    a = open_hicam_array_as_aligned_color_dataset(zarr_spool_location)

    shift_dict = calculate_channels_shift(a, reference_channel=anchor_channel)


    shift_medians = {
            0:None,
            1:None,
            2:None,
            3:None
        }

    for key in shift_dict:
        x = np.median(
            np.array(
            [x[0] for x in shift_dict[key]]
        )
        )
        x = 0 if x is np.nan else x
        y = np.median(
            np.array(
            [x[1] for x in shift_dict[key]]
        )
        )
        y = 0 if y is np.nan else y

        try:
            shift_medians[key] = (round(y),round(x))
        except Exception:
            shift_medians[key] = (0, 0)
    return shift_medians




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

