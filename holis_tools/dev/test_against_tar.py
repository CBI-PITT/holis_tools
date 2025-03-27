#
# /bil/users/awatson/miniconda3/envs/holis_tools/bin/python -i /bil/users/awatson/holis_tools/holis_tools/hicam_utils.py

import os, time, glob, dask
from dask.delayed import delayed
from distributed import Client

import subprocess

# subprocess.run(["ls", "-l"])


base = '/local_mount/space/buffer/LUN3/2024_01_30_dualHiCAM_sampleDataForCompression'
base_out = '/local_mount/space/buffer/LUN3/alan/comp_test'
# base_out = '/local_mount/space/buffer/LUN0/alan/comp_test'
# base_out = '/local_mount/space/buffer/LUN4/alan/comp_test'

# spool_files = (
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
# )

spool_files = glob.glob(base + '/*.fli')

# spool_files = [os.path.join(base,x) for x in spool_files]
compressions = (0,1,3,5,7,9)
compressions = (9,)
times = []

def run_cmd_list(cmd_list):
    out = os.system(cmd_list)
    return

archive_outs = []
for spool_file in spool_files:
    print(spool_file)
    file_name = os.path.split(spool_file)[-1]
    print(spool_file)
    file_name = file_name.replace(' ','_')
    print(spool_file)
    out_name = os.path.join(base_out,file_name + '.tar.zst')
    archive_outs.append(out_name)

compresses = (12,)*10
# compresses = (9,)*10
# compresses = (5,)*2
# compresses = (9,)*2
# compresses = (5,)*80
compress = 5
threads = 64
cmds = []
idx = 1
for compress in compresses:
    for spool_file, out_archive in zip(spool_files,archive_outs):
        # cmd = f'tar -I "zstd -{compress} -T{threads}" -cvpf {out_archive} "{spool_file}"'
        b, f = os.path.split(out_archive)
        new_path = os.path.join(b,f'12bit_zstd{compress}',str(idx),f)
        b, f = os.path.split(new_path)
        if not os.path.exists(b):
            os.makedirs(b, exist_ok=True)
        cmd = f'tar -I "zstd -{compress} -T{threads}" -cvpf {new_path} "{spool_file}"'
        print(cmd)
        # my_cmd_list = ["tar", "-I", f'"zstd -{compress} -T{threads}"', "-cvpf", f'"{out_archive}"', f'"{spool_file}"']
        tmp = delayed(os.system)(cmd)
        cmds.append(tmp)
        del tmp
    idx+=1

start = time.time()

# client = Client()
# out = client.compute(cmds)
out = dask.compute(cmds,num_workers=4)

stop = time.time()
total = stop-start
print(total)

#GB/s
print(f'Gigabits/s = {((254*len(compresses))/total)/0.125}')




'''
One at a time below
'''

import os, time, glob, dask
from dask.delayed import delayed
from distributed import Client

import subprocess

# subprocess.run(["ls", "-l"])


base = '/local_mount/space/buffer/LUN3/2024_01_30_dualHiCAM_sampleDataForCompression'
base_out = '/local_mount/space/buffer/LUN3/alan/comp_test'

# spool_files = (
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-088.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-088.fli',
#     'longRun3-y1-z0_HiCAM FLUO_1875-ST-272.fli',
#     'longRun4-y1-z1_HiCAM FLUO_1875-ST-272.fli'
# )

spool_files = glob.glob(base + '/*.fli')

# spool_files = [os.path.join(base,x) for x in spool_files]
compressions = (0,1,3,5,7,9)
compressions = (9,)
times = []

def run_cmd_list(cmd_list):
    out = os.system(cmd_list)
    return

archive_outs = []
for spool_file in spool_files:
    print(spool_file)
    file_name = os.path.split(spool_file)[-1]
    print(spool_file)
    file_name = file_name.replace(' ','_')
    print(spool_file)
    out_name = os.path.join(base_out,file_name + '.tar.zst')
    archive_outs.append(out_name)

# compresses = (5,5,5,5,5,5,5,5,5,5)
# compresses = (9,9,9,9,9,9,9,9,9,9)

compresses = (5,5)
compress = 5
threads = 64
cmds = []
idx = 1
start = time.time()
for compress in compresses:
    for spool_file, out_archive in zip(spool_files,archive_outs):
        # cmd = f'tar -I "zstd -{compress} -T{threads}" -cvpf {out_archive} "{spool_file}"'
        b, f = os.path.split(out_archive)
        new_path = os.path.join(b,f'12bit_zstd{compress}',str(idx),f)
        b, f = os.path.split(new_path)
        if not os.path.exists(b):
            os.makedirs(b, exist_ok=True)
        cmd = f'tar -I "zstd -{compress} -T{threads}" -cvpf {new_path} "{spool_file}"'
        print(cmd)
        # my_cmd_list = ["tar", "-I", f'"zstd -{compress} -T{threads}"', "-cvpf", f'"{out_archive}"', f'"{spool_file}"']
        print(f'Processing idx {idx}')
        tmp = os.system(cmd)
        # cmds.append(tmp)
        del tmp
    idx+=1



# client = Client()
# out = client.compute(cmds)
# out = dask.compute(cmds)

stop = time.time()
total = stop-start
print(total)

#GB/s
print(f'Gigabits/s = {((254*len(compresses))/total)/0.125}')
