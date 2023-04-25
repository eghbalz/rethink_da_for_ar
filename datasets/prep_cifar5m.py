
import numpy as np
import pickle

part = 0
data = np.load("/mnt/Blade1T/data/cifar5m/cifar5m_part{}.npz".format(part))

c, i2l, l2i = {}, {}, {}
for i, l in enumerate(data['Y']):
    if l not in c.keys():
        c[l] = 1
    else:
        c[l] += 1
    i2l[i] = l
    if l not in l2i.keys():
        l2i[l] = [i]
    else:
        l2i[l].append(i)

fname = open("/mnt/Blade1T/src/github/data_augmentation_robustness/datasets/files/l2i.pickle", "wb")
pickle.dump(l2i, fname, protocol=pickle.HIGHEST_PROTOCOL)

fname = open("/mnt/Blade1T/src/github/data_augmentation_robustness/datasets/files/i2l.pickle", "wb")
pickle.dump(i2l, fname, protocol=pickle.HIGHEST_PROTOCOL)

fname = open("/mnt/Blade1T/src/github/data_augmentation_robustness/datasets/files/c.pickle", "wb")
pickle.dump(c, fname, protocol=pickle.HIGHEST_PROTOCOL)

subl2i = {}
nsamp = 5000
for l in l2i.keys():
    ix = l2i[l]
    subix = np.random.choice(ix, nsamp, False)
    subl2i[l] = subix

fname = open("/mnt/Blade1T/src/github/data_augmentation_robustness/datasets/files/subl2i.pickle", "wb")
pickle.dump(subl2i, fname, protocol=pickle.HIGHEST_PROTOCOL)
