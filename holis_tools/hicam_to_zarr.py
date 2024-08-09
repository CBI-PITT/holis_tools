# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 18:43:51 2022

@author: alpha
"""

import argparse
import os
# import psutil

# psutil.virtual_memory()

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()
# print(args.accumulate(args.integers))

parser = argparse.ArgumentParser(description='''
                                 Convert a hicam file to chunked zarr
                                 ''')

positional = [
    ('input', str, '+', 'Input hicam file.'),
    ('output', str, 1, 'Output directory for zarr.')
]

optional = [

    (['-c', '--chunkSize'], int, 3, 'C', [128, 128, 128], 'store',
     'The chunk size for storing as zarr. Must be 3 integers for axes (ZYX).'),
    (['-f', '--framesAtOnce'], int, 1, 'F', [1024], 'store',
     '''The number of image frames to read in the hicam file with each group to be processed. Each group of 
     frames to be processed will consist of this many frames. Should be a multiple of the chunk sixe in Z.'''),
    (['-grp', '--groups'], int, 1, 'G', [100], 'store', 'Number of to be processed at once'),
    # (['-cpu'], int, 1, 'C', [os.cpu_count()], 'store', 'Number of cpus which are available'),
    # (['-mem'], int, 1, 'M', [int(psutil.virtual_memory().free / 1024 / 1024 / 1024 * 0.8)], 'store',
    #  'Available RAM in GB: default is 0.8x of free RAM'),
    # (['-tmp', '--tmpLocation'], str, 1, 'TMP', None, 'store',
    #  'Location for temp files --> high-speed local storage is suggested'),
    # (['-ft', '--fileType'], str, 1, 'FT', ['tif'], 'store',
    #  'File type for input --> Currently tif,tiff,jp2 and some nifti are supported'),
    # (['-s', '--scale'], float, 5, 'S', (1, 1, 1, 1, 1), 'store', '5-dim scale of the datasets (tczyx) in MICRONS'),

    # Compression options
    (['-cmp', '--compression'], str, 1, 'CMP', ['zstd'], 'store', 'Compression method (zstd supported)'),
    (['-cl', '--clevel'], int, 1, 'CMP', [5], 'store', 'Compression level : Integer 0-9 zstd (default 5)'),

    # # Optional arguments for OME-ZARR OMERO metadata
    # (['-ch', '--channelLabels'], str, '*', 'CH', [], 'store', 'A label for each channel'),
    # (['-clr', '--colors'], str, '*', 'CLR', [], 'store',
    #  'Color for each channel: by default colors are repeatedly assigned as [green, red, purple, blue,...]'),
    # (['-win', '--windowLabels'], int, '*', 'WIN', [], 'store',
    #  'START END MIN MAX defining the LUT representation of each channel'),
    # (['-n', '--name'], str, 1, 'NAM', [], 'store', 'Name of the dataset'),
    # (['-z', '--defaultZ'], int, 1, 'dfZ', [], 'store', 'Default Z-Layer to be displayed during visualization'),
    # (['-dst', '--downSampleType'], str, 1, 'DSM', ['mean'], 'store',
    #  'Down sample method. Options are mean and max (default: mean'),

]
# switch = [
#     (['-v', '--verbose'], 0, 'count', 'Verbose output : additive more v = greater level of verbosity'),
#     (['-vzw', '--verify_zarr_write'], False, 'store_true', 'Immediately verify each chunk written to disk.'),
#     (['-sk', '--skip'], False, 'store_true', 'skip resolution level if it already exist'),
#     (['-st', '--stopBuild'], False, 'store_true',
#      'Immediately stop building multiscale NGFF after initializing builder class- only used for development purposes'),
#     (['-df', '--directToFinalChunks'], False, 'store_true',
#      'Use final chunks for all multiscales except full resolution'),
#     (['-bt', '--buildTmpCopyDestination'], False, 'store_true',
#      'Use final chunks for all multiscales except full resolution'),
#
#     # Compression opts
#     # Force users to choose --lossy for this to be used, otherwise default to lossless
#     (['-los', '--lossy'], False, 'store_true',
#      'Use lossy compression, this only matters if using jpegxl and it MUST be selected for lossy compression to be enabled'),
#     (['-losms', '--lossyms'], False, 'store_true',
#      'Use lossy compression for multiscales, this only matters if using jpegxl and it MUST be selected for lossy compression to be enabled'),
#     # (['-lms', '--lossyMultiscale'], False,'store_true','Use lossy compression, only for multiscale data'),
#
#     # Features relevant for zarr_stores (https://github.com/CBI-PITT/zarr_stores)
#     (['-wdo', '--writeDirectOff'], False, 'store_true',
#      'If selected, sharded stores from the zarr_stores package will write data as a NestedDirectoryStore before consolodating to sharded after completion'),
# ]

for var, v_type, nargs, v_help in positional:
    parser.add_argument(var, type=v_type, nargs=nargs, help=v_help)

for var, v_type, nargs, metavar, default, action, v_help in optional:
    parser.add_argument(*var, type=v_type, nargs=nargs, metavar=metavar, default=default, action=action, help=v_help)

# for var, default, action, v_help in switch:
#     parser.add_argument(*var, default=default, action=action, help=v_help)

# args = parser.parse_args()
# print(args)
# print(args.input)
# print(args.output)
# print(args.mem)
# print(args.cpu)
# print(args.finalChunkSize)
# print(args.origionalChunkSize)


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

    hicam_file = args.input[0]
    zarr_location = args.output[0]
    compressor_type = args.compression[0]
    compressor_level = args.clevel[0]
    chunk_depth = args.chunkSize[0]
    chunk_lat = args.chunkSize[1] if len(args.chunkSize) > 1 else chunk_depth
    frames_at_once = args.framesAtOnce[0]
    groups = args.groups[0]

    # print(dir())
    # print(globals())

    # for attr, value in dict(dir()):
    #     print(f'{attr}=={value}')

    # from holis_tools.hicam_utils import send_hicam_to_zarr_par_read_groups

    #
    # send_hicam_to_zarr_par_read_groups(hicam_file, zarr_location, compressor_type='zstd', compressor_level=5,
    #                                        shuffle=1, chunk_depth=128, chunk_lat=128, frames_at_once=1024, groups=100):


# Display all vars
all_variables = dir()
for name in all_variables:

    # Print the item if it doesn't start with '__'
    if not name.startswith('__') and not name == 'optional' and not name == 'positional' and not name == 'parser':
        myvalue = eval(name)
        # print(name, "is", type(myvalue), "and is equal to ", myvalue)
        print(f'{name} == {type(myvalue)} == {myvalue}')