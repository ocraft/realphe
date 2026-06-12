from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Optional

import numpy as np
from numpy._typing import _Shape
import pandas as pd


class SharedArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''

    def __init__(self, array: np.ndarray, name: Optional[str] = None):
        '''
        Creates the shared memory and copies the array therein
        '''
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(name=name, create=True, size=array.nbytes)

        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape

        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )

        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        '''
        Returns a new copy of the array stored in shared memory.
        '''
        return np.copy(self.read())

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()

    def close(self):
        self._shared.close()

    @property
    def shape(self) -> _Shape:
        return self._shape


class SharedDataFrame:
    '''
    Wraps a pandas dataframe so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''

    def __init__(self, data: pd.DataFrame | SharedArray, index=None, columns=None):
        '''
        Creates the shared memory and copies the dataframe therein
        '''
        if isinstance(data, pd.DataFrame):
            self._values = SharedArray(data.values)
            self._index = data.index
            self._columns = data.columns
        else:
            self._values = data
            self._index = index
            self._columns = columns

    def read(self) -> pd.DataFrame:
        '''
        Reads the dataframe from the shared memory
        without unnecessary copying.
        '''
        return pd.DataFrame(
            self._values.read(),
            index=self._index,
            columns=self._columns
        )

    def array(self) -> SharedArray:
        return self._values

    def copy(self):
        '''
        Returns a new copy of the dataframe stored in shared memory.
        '''
        return pd.DataFrame(
            self._values.copy(),
            index=self._index,
            columns=self._columns
        )

    def __setitem__(self, key, value):
        df = self.read()
        df.loc[key] = value

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._values.unlink()

    def map(self, func: Callable[[pd.DataFrame], pd.DataFrame]):
        df = func(self.read())
        self._values.read()[:] = df.values[:]

        del df

    @property
    def shape(self) -> _Shape:
        return self._values._shape
