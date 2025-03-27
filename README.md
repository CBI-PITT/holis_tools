# holis_tools

## This repository offers tools to deal with access to holis_data

#### Installing:

```bash
# Clone the repo
cd /dir/of/choice
git clone https://github.com/CBI-PITT/holis_tools.git

# Create a virtual environment
# This assumes that you have miniconda or anaconda installed
conda create -n holis_tools python=3.12 -y

# Activate environment and install holis_tools
conda activate holis_tools
pip install -e /dir/of/choice/holis_tools
```



##### <u>spool_reader:</u>

###### Description:

This tool enables loading zyla spool_files representing 1yz strip into into memory as a numpy array. Spool files can be read from a directory natively produced by the camera or from a compressed zip file produced by the library compression_tools found here: https://github.com/CBI-PITT/compression_tools/tree/main/compression_tools

###### Usage example:

```python
from compression_tools.alt_zip import alt_zip
from holis_tool.zyla_spool_reader import spool_set_interpreter
import numpy as np

# Location of zip file or directory contatining spool files
test_spool_zip = r'/zip/file/writted/using/compression_tools/containing/spool/files.zip'

# Instantiate class to manage spool files (Only metadata is loaded from disk)
a = spool_set_interpreter(test_spool_zip)
a.dtype
a.spool_shape

# Load only the 100th spool file from disk
hundredth_spool_file = a[100]
hundredth_spool_file.shape == a.spool_shape #True

# Read all data from disk and assemble a single numpy array
b = a.assemble()
b.shape
b.dtype

# Each camera frame is along the first axis
first_frame = b[0]
hundredth_frame = b[99]

```

Other modules are incomplete and not documented
