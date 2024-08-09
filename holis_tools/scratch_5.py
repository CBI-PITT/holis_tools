
import os
import numpy as np
from holis_tools.hicam_utils import read_header
file = r'Z:\hillman\test_run_small\longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
file = r'/CBI_FastStore/hillman/test_run_small/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
out_location = os.path.split(file)[0] + '/test_out' ## TEMP FOR TESTING

spool_file = file

header_info, _ = read_header(spool_file)
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
        num_frames_remainder = data_size % pixelInFrame_bit8
        print(f'{num_frames_remainder=}')
        how_many_frames = data_size // pixelInFrame_bit8
        print(f'{how_many_frames=}')

chunk_shape = (how_many_frames, header_info['y'], header_info['x'])

outputa = np.zeros(chunk_shape, 'uint16')
outputb = np.zeros(chunk_shape, 'uint16')
outputc = np.zeros(chunk_shape, 'uint16')

with open(spool_file, 'rb') as f:
    size_of_file = f.tell()
    print(f'{size_of_file=}')
    print(f'Reading {how_many_frames} frames')
    f.seek(header_info['headerLength'])
    multi_frame = f.read(how_many_frames * pixelInFrame_bit8)


def read_uint12_a(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return array

def read_uint12_b(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = (lst_uint8 << 4) + (np.bitwise_and(15, mid_uint8))
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return array

### THIS ONE WORKS FOR HICAM ###
def read_uint12_c(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = ((mid_uint8 & 0x0F) << 8) | fst_uint8
    snd_uint12 = (lst_uint8 << 4) | ((mid_uint8 & 0xF0) >> 4)
    array = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
    return array


print(f'Forming Array')
for idx in range(1000): #how_many_frames
    print(f'Converting {idx} of {how_many_frames}')
    where_to_start = idx * pixelInFrame_bit8
    data = multi_frame[where_to_start:where_to_start + pixelInFrame_bit8]
    # Data to uint16 where uint12 values have been scaled to uint16 values
    # uint16 scaling is important for downstream manipulation as float or for visualization accuracy
    canvasa = read_uint12_a(data)
    canvasb = read_uint12_b(data)
    canvasc = read_uint12_c(data)
    # canvas = read_uint12(data, coerce_to_uint16_values=False)
    outputa[idx] = canvasa.reshape((header_info['y'], header_info['x']))
    outputb[idx] = canvasb.reshape((header_info['y'], header_info['x']))
    outputc[idx] = canvasc.reshape((header_info['y'], header_info['x']))

from skimage import io
imagea = outputa[500,:,:]
imageb = outputb[500,:,:]
imagec = outputc[500,:,:]
out_filea = os.path.split(file)[0] + r'/outa.tiff'
out_fileb = os.path.split(file)[0] + r'/outb.tiff'
out_filec = os.path.split(file)[0] + r'/outc.tiff'
io.imsave(out_filea, imagea)
io.imsave(out_fileb, imageb)
io.imsave(out_filec, imagec*16)
