


# ON BIL:
#
# /bil/users/awatson/miniconda3/envs/holis_tools/bin/python -i /bil/users/awatson/holis_tools/holis_tools/hicam_utils.py

import os, time, glob

base = '/bil/proj/rf1hillman/2024_01_30_dualHiCAM_sampleDataForCompression'

# spool_files = (
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
# )

spool_files = glob.glob(base + '/*.fli')

# spool_files = [os.path.join(base,x) for x in spool_files]
zarr_locations = [os.path.join(r'/bil/users/awatson/holis',os.path.split(x)[-1] + '_ZARR_OUT') for x in spool_files]


def run(spool_files=spool_files, zarr_locations=zarr_locations):
    start = time.time()
    for spool_file, zarr_location in zip(spool_files, zarr_locations):
        send_hicam_to_zarr_par_read_once(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=128)
    stop = time.time()
    print(f'Total time = {stop}')


run(spool_files=spool_files, zarr_locations=zarr_locations)


if __name__ == '__main__':
    run()