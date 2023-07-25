import sys
sys.getdefaultencoding()
import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = '../data_utils/dslr_9_1.pkl'
file = open(path,'rb')
inf = pickle.load(file,encoding='iso-8859-1')       #读取pkl文件的内容
# print(inf)
#fr.close()
inf=str(inf)
obj_path = '../data/office31/dslr_val_9_1.txt'
ft = open(obj_path, 'w')
ft.write(inf)
