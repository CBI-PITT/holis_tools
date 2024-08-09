


# ON BIL:
#
# /bil/users/awatson/miniconda3/envs/holis_tools/bin/python -i /bil/users/awatson/holis_tools/holis_tools/hicam_utils.py

import os, time, glob

base = '/bil/proj/rf1hillman/2024_01_30_dualHiCAM_sampleDataForCompression'
base = '/local_mount/space/buffer/LUN3/2024_01_30_dualHiCAM_sampleDataForCompression'

# spool_files = (
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
# )

spool_files = glob.glob(base + '/*.fli')

# spool_files = [os.path.join(base,x) for x in spool_files]
compressions = (0,1,3,5,7,9)

times = []
for comp in compressions:
    start = time.time()
    for spool_file in spool_files:
        zarr_location = os.path.join(r'/local_mount/space/buffer/LUN3/alan',str(comp),os.path.split(spool_file)[-1] + '_ZARR_OUT')
        send_hicam_to_zarr_par_read_once(spool_file, zarr_location, compressor_type='zstd', compressor_level=comp,
                                         shuffle=1, chunk_depth=128, frames_at_once=128)
    stop = time.time()
    total = stop-start
    times.append(total)



import numcodecs
from numcodecs.blosc import Blosc
from numcodecs import register_codec
register_codec(Blosc)
with open(spool_file, 'rb') as f:
    data = f.read()

compressor = Blosc(
        cname='zstd',
        clevel=5,
        shuffle=Blosc.SHUFFLE,
        blocksize=0
    )
numcodecs.blosc.set_nthreads(30)

buffer_size = (2147483647-1)//2
out = bytes()
for start in range(0,len(data),buffer_size):
    stop = start + buffer_size
    tmp = compressor.encode(data[start:stop])
    out += tmp










import numcodecs
from numcodecs.blosc import Blosc
from numcodecs import register_codec
register_codec(Blosc)


compressor = Blosc(
        cname='zstd',
        clevel=5,
        shuffle=Blosc.SHUFFLE,
        blocksize=0
    )
numcodecs.blosc.set_nthreads(30)

buffer_size = (2147483647-1)//2
orig_len = 0
compress_len = 0
for spool_file in spool_files:
    out = bytes()
    with open(spool_file, 'rb') as f:
        print(f'Reading file: {spool_file}')
        data = f.read()
    orig_len += len(data)
    for start in range(0, len(data), buffer_size):
        stop = start + buffer_size
        tmp = compressor.encode(data[start:stop])
        compress_len += len(tmp)
        del tmp
    print(f'Compress Ratio = {compress_len/orig_len}')





# ON YODA:
#
# /bil/users/awatson/miniconda3/envs/holis_tools/bin/python -i /bil/users/awatson/holis_tools/holis_tools/hicam_utils.py

import os, time, glob, dask

base = '/local_mount/space/buffer/LUN3/2024_01_30_dualHiCAM_sampleDataForCompression'

# spool_files = (
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
# )

spool_files = glob.glob(base + '/*.fli')

# spool_files = [os.path.join(base,x) for x in spool_files]
compressions = (0,1,3,5,7,9)
compressions = (7,)
times = []

from distributed import Client
with Client() as client:
    for comp in compressions:
        start = time.time()
        for spool_file in spool_files:
            zarr_location = os.path.join(r'/local_mount/space/buffer/LUN3/alan',str(comp),os.path.split(spool_file)[-1] + '_ZARR_OUT')
            send_hicam_to_zarr_par_read_groups_par(spool_file, zarr_location, compressor_type='zstd', compressor_level=comp,
                                             shuffle=1, chunk_depth=128, frames_at_once=128, groups=100)
        stop = time.time()
        total = stop-start
        times.append(total)


def run():
    # ON YODA:
    #
    # /bil/users/awatson/miniconda3/envs/holis_tools/bin/python -i /bil/users/awatson/holis_tools/holis_tools/hicam_utils.py

    import os, time, glob, dask

    base = '/local_mount/space/buffer/LUN3/2024_01_30_dualHiCAM_sampleDataForCompression'

    # spool_files = (
    #     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
    #     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
    #     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
    #     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
    # )

    spool_files = glob.glob(base + '/*.fli')

    # spool_files = [os.path.join(base,x) for x in spool_files]
    compressions = (0,1,3,5,7,9)
    # compressions = (5,)
    compressions = (9,9,9,9,9,9,9,9,9,9) #'Gigabits/s = 6.086620618140829'
    compressions = (7,7,7,7,7,7,7,7,7,7) #'Gigabits/s = 8.060902494602352'
    # compressions = (5,5,5,5,5,5,5,5,5,5) #'Gigabits/s = 9.590964553455757'
    # compressions = (3,3,3,3,3,3,3,3,3,3) #'Gigabits/s = 11.031535271293594'
    # compressions = (1,1,1,1,1,1,1,1,1,1) #'Gigabits/s = 12.196180515670102'

    '''
    Below will compress 10 * 254GB of data represented by 4 distinct .fli files
    '''
    times = []

    computing = []
    from distributed import Client
    # client = Client(n_workers=None) #Defaults to 24, I think
    # client = Client(n_workers=4)
    with Client() as client:
    idx = 1
    for comp in compressions:
        start = time.time()
        for spool_file in spool_files:
            zarr_location = os.path.join(r'/local_mount/space/buffer/LUN3/alan/comp_test',str(idx),str(comp),os.path.split(spool_file)[-1] + '_ZARR_OUT')
            out = send_hicam_to_zarr_par_read_groups_par(spool_file, zarr_location, compressor_type='zstd', compressor_level=comp,
                                             shuffle=1, chunk_depth=128, frames_at_once=128, groups=100)
            computing.append(out)
            del out
        idx += 1
    while any([x.status != 'finished' for y in computing for x in y]):
        time.sleep(1)
    stop = time.time()
    total = stop-start
    times.append(total)
    client.close()
    print(f'Gigabits/s = {((254*len(compressions))/times[0])/0.125}')
    return times



'''
Testing tar zstd from raw 12bit data
Example cmd:

tar -I "zstd -9 -T64" -cvpf /local_mount/space/buffer/LUN3/alan/comp_test/12bit_zstd9/1/longRun3-y1-z0_HiCAM_FLUO_1875-ST-088.fli.tar.zst 
"/local_mount/space/buffer/LUN3/2024_01_30_dualHiCAM_sampleDataForCompression/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli"

All tests use 64 threads
10 runs over all 4 files, 40 files in total
254 GB * 10 runs = 2.5TB

All done on Yoda
'''

#1 at a time:
# ZSTD5 'Gigabits/s = 12.874920065229482', 1.4TB, 1.610 GB/s
# ZSTD9 'Gigabits/s = 9.729085796092082',  1.3TB, 1.216 GB/s

#Parallel (4 at a time)
# ZSTD5 'Gigabits/s = 14.221630482909852', 1.4TB, 1.778 GB/s
# ZSTD9 'Gigabits/s = 15.253226319840126', 1.3TB, 1.907 GB/s

#Parallel (24 at a time)
# ZSTD5 'Gigabits/s = 12.877924082789084', 1.4TB, 1.610 GB/s
# ZSTD9 'Gigabits/s = 11.668648575058334', 1.3TB, 1.459 GB/s

'''
80 runs over all 4 files, 320 files in total
254 GB * 80 runs = 20.3 TB

On Yoda
'''
# ZSTD5  'Gigabits/s = 13.162588861013717', 11 TB, 1.645 GB/s

'''
Only 8 files, 4 at a time
2 runs over all 4 files, 8 files in total
254 GB * 2 runs = 0.5TB
'''
#Parallel (4 at a time)
# ZSTD5 'Gigabits/s = 35.111557557512334', 0.5TB, 4.39 GB/s
# ZSTD9 'Gigabits/s = 18.586426716014394', 0.5TB, 2.32 GB/s


'''
To Zarr using hicam_tools

10 runs over all 4 files, 40 files in total
254 GB * 10 runs = 2.5TB
'''
# ZSTD9, 128^3 chunks, 'Gigabits/s = 6.086620618140829'
# ZSTD7, 128^3 chunks, 'Gigabits/s = 8.060902494602352'
# ZSTD5, 128^3 chunks, 'Gigabits/s = 9.590964553455757'
# ZSTD3, 128^3 chunks, 'Gigabits/s = 11.031535271293594'
# ZSTD1, 128^3 chunks, 'Gigabits/s = 12.196180515670102'
