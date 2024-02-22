



spool_file = r'H:\globus\pitt\bil\hillman\spool_examples\hicam\2023_04_24_HiCAMTest_sampleDataSet\hicam-scan-ROIb15_HiCAM FLUO_1875-ST-272.fli'
spool_file = r'/CBI_Hive/globus/pitt/bil/hillman/spool_examples/hicam/2023_04_24_HiCAMTest_sampleDataSet/hicam-scan-ROIb15_HiCAM FLUO_1875-ST-272.fli'
spool_file = r'/h20/hicam/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'


# read_header(spool_file)

zarr_location = spool_file + '_zarr_out'
send_hicam_to_zarr(spool_file,zarr_location,compressor_type='zstd', compressor_level=5, shuffle=1)
