"""Check the structure of E.T. smplh pickle files."""
import pickle

path = '/transfer/et-data/smplh/2011_009tNfQRd4o_00000_00001.pkl'
d = pickle.load(open(path, 'rb'))

for k, v in d.items():
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
