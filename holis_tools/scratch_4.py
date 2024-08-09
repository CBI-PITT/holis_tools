
file = r'Z:\hillman\test_run_small\longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
file = r'/CBI_FastStore/hillman/test_run_small/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
from holis_tools.hicam_utils import read_header
import holis_tools
import os
read_header(file)
out_location = os.path.split(file)[0] + '/test_out3' ## TEMP FOR TESTING
holis_tools.hicam_utils.send_hicam_to_zarr_par(file,out_location,compressor_type='zstd', compressor_level=5, shuffle=1, chunk_depth=512, frames_at_once=1024, cube_chunk=None)

import os
from skimage import io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

file = r'Z:\hillman\test_run_small\longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
file = r'/CBI_FastStore/hillman/test_run_small/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
out_location = os.path.split(file)[0] + '/test_out3' ## TEMP FOR TESTING
from zarr_stores.h5_nested_store import H5_Nested_Store
import zarr

a = zarr.open(H5_Nested_Store(out_location))

image = a[20000:20100,:,:]
image = a[:]
io.imsave(os.path.split(file)[0] + r'/out_zarr_3.tiff', image[0:2000])

q = holis_tools.hicam_utils.read_data_file(file)



import os
from skimage import io

file = r'Z:\hillman\test_run_small\longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
file = r'/CBI_FastStore/hillman/test_run_small/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
out_location = os.path.split(file)[0] + '/test_out2' ## TEMP FOR TESTING


a = zarr.open(H5_Nested_Store(out_location))

image = a[20000,:,:]
io.imsave(os.path.split(file)[0] + r'\out.tiff', image)

## Read whole file and convert to 16bit
import os
from skimage import io
from holis_tools.hicam_utils import read_data_file
file = r'Z:\hillman\test_run_small\longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
file = r'/CBI_FastStore/hillman/test_run_small/longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli'
out_location = os.path.split(file)[0] + '/test_out' ## TEMP FOR TESTING
q = read_data_file(file)

image = q[20000,:,:]
out_file = os.path.split(file)[0] + r'/out.tiff'
io.imsave(out_file, image)