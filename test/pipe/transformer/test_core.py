import xarray as xr

from realphe.pipe.transformer.core import fit_transform
from realphe.pipe.transformer.imputer import FillNa
from realphe.pipe.transformer.scaler import StandardScaler

# TODO dask transformers
# def test_out_of_core_transform(tmp_path, dataset3d):
#     transformers = [
#         FillNa(dataset3d.feature),
#         StandardScaler(dataset3d.feature),
#         FillNa(dataset3d.feature),
#     ]
#     transform(dataset3d.chunk({'sample': 1}), transformers, out_path=tmp_path / 'test.nc').close()

#     xr.testing.assert_identical(
#         xr.open_dataset(tmp_path / 'test.nc'),
#         transform(dataset3d, transformers)
#     )
