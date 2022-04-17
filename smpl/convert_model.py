
import pickle

fname = 'models/basicModel_f_lbs_10_207_0_v1.1.0.pkl'

with open(fname, 'rb') as f:
  dd = pickle.load(f, encoding='latin1')

print(type(dd))
print(dd)

