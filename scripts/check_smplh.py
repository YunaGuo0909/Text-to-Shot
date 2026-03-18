"""Check the structure of E.T. smplh pickle files."""
import pickle
import numpy as np

path = '/transfer/et-data/smplh/2011_009tNfQRd4o_00000_00001.pkl'
d = pickle.load(open(path, 'rb'))
print("Type:", type(d))

if isinstance(d, dict):
    print("Keys:", list(d.keys()))
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")
else:
    print("Not a dict, type:", type(d))
