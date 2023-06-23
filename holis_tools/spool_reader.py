

from compression_tools.alt_zip import alt_zip
import configparser
import numpy as np

# test_spool_zip = r'H:\globus\pitt\bil\hillman\spool_examples\COMPRESSED_CLEV5_Planes_secondColor_run12__z01_y12_Exc_488nm_660nm.zip'
#
# spool_set = alt_zip(test_spool_zip)

# with open(r'H:\globus\pitt\bil\hillman\spool_examples\Planes_secondColor_run12__z01_y12_Exc_488nm_660nm\acquisitionmetadata.ini','r') as f:
#     print(f.read())

class spool_set_interpreter:

    def __init__(self,compression_tools_zip_file):
        self.compression_tools_zip_file = compression_tools_zip_file
        self.location = compression_tools_zip_file
        self.spool_set = alt_zip(self.location)

        self._list_spool_files()
        self._get_acquisitionparameters_str()
        self._get_config()
        self._extract_config_values()


    @property
    def entries(self):
        return self.spool_set.entries

    def _list_spool_files(self):
        self.spool_files = sorted(
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
        self.acquisition_metadata['height'] = config.getint('data', 'AOIHeight')
        self.acquisition_metadata['width'] = config.getint('data', 'AOIWidth')
        self.acquisition_metadata['stride'] = config.getint('data', 'AOIStride')
        dtype = config.get('data', 'PixelEncoding')

        if dtype == 'Mono16':
            dtype = np.dtype('uint16')
        elif dtype == 'Mono8':
            dtype = np.dtype('uint8')

        self.acquisition_metadata['dtype'] = dtype
        self.dtype = dtype

        self.nbytes = config.getint('data', 'ImageSizeBytes')
        self.acquisition_metadata['nbytes'] = self.nbytes

        self.acquisition_metadata['images'] = config.getint('multiimage', 'ImagesPerFile')

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
        array = np.frombuffer(self.spool_set[spool_file_name],dtype=a.dtype)
        return np.reshape(array, self.spool_shape)

    def __getitem__(self,key):
        if isinstance(key,str):
            assert key in self.spool_files, 'Must be a spool file in self.spool_files OR integer index in self.spool_files'
            return self._load_spool_file(key)
        elif isinstance(key,int):
            return self._load_spool_file(self.spool_files[key])

    def __iter__(self):
        yield from (self[x] for x in range(len(self.spool_files)))

    def __contains__(self, item):
        return item in self.spool_files


# a = spool_set_interpreter(test_spool_zip)
# a._load_spool_file('8970000000spool.dat')

