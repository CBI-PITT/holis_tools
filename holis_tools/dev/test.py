



spool_file = r'H:\globus\pitt\bil\hillman\spool_examples\hicam\2023_04_24_HiCAMTest_sampleDataSet\hicam-scan-ROIb15_HiCAM FLUO_1875-ST-272.fli'
spool_file = r'/CBI_Hive/globus/pitt/bil/hillman/spool_examples/hicam/2023_04_24_HiCAMTest_sampleDataSet/hicam-scan-ROIb15_HiCAM FLUO_1875-ST-272.fli'
spool_file = r'/h20/hicam/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'


# read_header(spool_file)

zarr_location = spool_file + '_zarr_out'
send_hicam_to_zarr(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=1024)





spool_file = r'/h20/hicam/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
for ii in get_start_stop_reads_for_frame_groups(spool_file, header_info=None, frames_at_once=128):
    print(ii)



spool_file = r'/h20/hicam/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
zarr_location = spool_file + '_zarr_out'
# send_hicam_to_zarr_par(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=256)
send_hicam_to_zarr_par_read_once(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=256)



ON BIL:

/bil/users/awatson/miniconda3/envs/holis_tools/bin/python -i /bil/users/awatson/holis_tools/holis_tools/hicam_utils.py

spool_file = r'/bil/proj/rf1hillman/2024_01_30_dualHiCAM_sampleDataForCompression/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
zarr_location = r'/bil/users/awatson/holis/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli_ZARR_OUT'
# send_hicam_to_zarr_par(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=256)
send_hicam_to_zarr_par_read_once(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=128, frames_at_once=256)

#
# import time
# start = time.time()
# f = open(spool_file, 'rb')
# data = f.read()
# f.close()
# stop = time.time()
# print(stop-start)
