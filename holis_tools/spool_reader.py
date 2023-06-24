

from compression_tools.alt_zip import alt_zip
import configparser
import numpy as np
import io
import matplotlib
import warnings



test_spool_zip = r'H:\globus\pitt\bil\hillman\spool_examples\COMPRESSED_CLEV5_Planes_secondColor_run12__z01_y12_Exc_488nm_660nm.zip'

# spool_set = alt_zip(test_spool_zip)

# with open(r'H:\globus\pitt\bil\hillman\spool_examples\Planes_secondColor_run12__z01_y12_Exc_488nm_660nm\acquisitionmetadata.ini','r') as f:
#     print(f.read())

class spool_set_interpreter:

    def __init__(self,compression_tools_zip_file):
        self.compression_tools_zip_file = compression_tools_zip_file
        self.location = compression_tools_zip_file
        self.spool_set = alt_zip(self.location)

        self._what_spool_format()
        self.spool_files = tuple(self._get_spool_names_in_order()) # In order
        self._get_acquisitionparameters_str()
        self._get_config()
        self._extract_config_values()


    @property
    def entries(self):
        return self.spool_set.entries

    def _what_spool_format(self):
        if 'Spooled files.sifx' in self.entries:
            format = 'zyla'
        else:
            raise TypeError("Unknown or unsupported spool file format")

        self.format = format

    def _list_spool_files(self):
        return sorted(
            tuple(
            [x for x in self.entries if '0spool.dat' in x]
        )
        )
    def _get_acquisitionparameters_str(self):
        ini = self.spool_set['acquisitionmetadata.ini']
        ini = ini.decode('UTF-8-sig')  # Encoding for acquisitionmetadata.ini
        self.acquisitionparameters_str = ini

    def _get_config(self):
        buf = io.StringIO(self.acquisitionparameters_str)
        self.config = configparser.ConfigParser()
        self.config.read_file(buf)

    def _extract_config_values(self):
        self.acquisition_metadata = {}
        # ini info
        self.acquisition_metadata['height'] = self.config.getint('data', 'AOIHeight')
        self.acquisition_metadata['width'] = self.config.getint('data', 'AOIWidth')
        self.acquisition_metadata['stride'] = self.config.getint('data', 'AOIStride')
        dtype = self.config.get('data', 'PixelEncoding')

        if dtype == 'Mono16':
            dtype = np.dtype('uint16')
        elif dtype == 'Mono8':
            dtype = np.dtype('uint8')

        self.acquisition_metadata['dtype'] = dtype
        self.dtype = dtype

        self.spool_nbytes = self.config.getint('data', 'ImageSizeBytes')
        self.acquisition_metadata['nbytes'] = self.spool_nbytes

        self.acquisition_metadata['images'] = self.config.getint('multiimage', 'ImagesPerFile')

        numDepths = self.acquisition_metadata['height']
        numLatPix = self.acquisition_metadata['stride'] // 2
        imageBytes = self.acquisition_metadata['nbytes']
        numFramesPerSpool = self.acquisition_metadata['images']
        startIndex = self.acquisition_metadata['nbytes']
        imageSize = self.acquisition_metadata['nbytes']

        numRows = numDepths + 2

        if numDepths % 2:  # if there is an odd number of rows ->  KPEDIT - odd rows means 1 less column for some reason
            numRows = numDepths + 1

        numColumns = numLatPix

        self.spool_shape = (numFramesPerSpool, numRows, numColumns)

    def _load_spool_file(self,spool_file_name):
        array = np.frombuffer(self.spool_set[spool_file_name],dtype=self.dtype)
        return np.reshape(array, self.spool_shape)

    def __getitem__(self,key):
        if isinstance(key,str):
            assert key in self.spool_files, 'Must be a spool file in self.spool_files OR integer index in self.spool_files'
            return self._load_spool_file(key)
        elif isinstance(key,int):
            return self._load_spool_file(self.spool_files[key])

    def __iter__(self):
        yield from (self[x] for x in range(len(self)))

    def __contains__(self, item):
        return item in self.spool_files

    def __len__(self):
        return len(self.spool_files)


    def _get_spool_names_in_order(self):
        '''
        Spool files are ordered sequentially 0,1,2,...,201,202,203,... but are named as a reverse number padded to
        10 digits (0000000000,1000000000,20000000000,...,1020000000,2020000000,3020000000,...) + spool.dat
        '''
        spool_files = self._list_spool_files()
        misses=0
        for idx in range(len(spool_files)):
            # Convert index to string, pad with zeros to 10 digits and reverse
            tmp = str(idx+misses).zfill(10)[::-1]
            tmp = f'{tmp}spool.dat'
            if tmp in spool_files:
                yield tmp
            else:
                warnings.warn(f"{tmp} not located in spool directory")
                misses += 1

    def assemble(self):
        axis_0_shape = self.spool_shape[0]
        canvas = np.zeros((axis_0_shape*len(self),*self.spool_shape[1:]), dtype=self.dtype)
        for idx,spool_file in enumerate(self):
            start = idx*axis_0_shape
            stop = start + axis_0_shape
            canvas[start:stop] = spool_file
        return canvas


a = spool_set_interpreter(test_spool_zip)
b = a.assemble()

c = b[:,100]
import skimage
import matplotlib.pyplot as plt
skimage.io.imshow(c*100)
plt.show()


