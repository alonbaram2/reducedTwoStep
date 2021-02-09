import pandas as pd
from scipy import io as sio
from os.path import dirname, abspath, join

data_dir = join(dirname(dirname(abspath(__file__))), 'bhv_data')

data = pd.read_pickle('data.pkl')
data_dict = {col_name : data[col_name].values for col_name in data.columns.values}
sio.savemat('data.mat', {'data':data_dict})

fits = pd.read_pickle('fits.pkl')

# deal with the multi-indexing in "fits" - get rid of the the param_ranges. 
fits = fits.drop(columns = 'param_ranges')
fits = fits.rename(columns = {'params':''})
fits.columns = [''.join(col).strip() for col in fits.columns.values]

# save fits
fits_dict = {col_name : fits[col_name].values for col_name in fits.columns.values}
sio.savemat('fits.mat', {'fits':fits_dict})

