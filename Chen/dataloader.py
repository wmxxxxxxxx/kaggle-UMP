import torch
import numpy as np


# a = np.array([[1,2,3,4],[2,3,4,5],[4,3,4,4],[3,4,3,6],[5,6,7,8]])
# random = np.arange(3)
# np.random.shuffle(random)
# a[:3] = a[random]
# print(a)



class DataLoader(object):
    def __init__(self, batch_size=1024):
        self.data_set = np.load('./train.npy', allow_pickle=True)
        self.count_list = np.load('./count.npy', allow_pickle=True)
        self.batch_list = []
        self.batch_data = []
        for i in range(len(self.count_list)):
            mod = int(self.count_list[i]) % int(batch_size)
            mul = int(int(self.count_list[i] - mod) / int(batch_size))
            for j in range(mul):
                self.batch_list.append(batch_size)
            if mod != 0:
                self.batch_list.append(mod)
        self.batch_num = len(self.batch_list)
        self.iteration = 0

    def shuffle(self, in_arr, count):
        out_arr = in_arr
        for i in range(len(count)):
            random = np.arange(count[i])
            out_arr[:count[i]] = out_arr[random]
        return out_arr

    def initialize_data(self):
        self.batch_data = []
        self.iteration = 0
        temp_data = self.shuffle(self.data_set, self.count_list)
        num = 0
        for i in range(len(self.batch_list)):
            self.batch_data.append(temp_data[num:num+self.batch_list[i]])
            num = num + self.batch_list[i]

    def get_batch_data(self):
        if self.iteration == self.batch_num:
            self.initialize_data()
        batch_data = np.array(self.batch_data[self.iteration])
        self.iteration += 1
        batch_data = torch.from_numpy(batch_data)
        investment_id = batch_data[:, 1:2]
        feature = batch_data[:, 3:]
        #feature = feature.unsqueeze(0)
        target = batch_data[:, 2:3]
        return investment_id, feature, target

if __name__=='__main__':
    Dataset = DataLoader()
    Dataset.initialize_data()
    print(Dataset.batch_num)
    investment_id, feature, target = Dataset.get_batch_data()
    print(feature.shape)
    print(investment_id.shape)
    print(target.shape)
    # a = np.load('./train.npy', allow_pickle=True)
    # b = np.load('./count.npy', allow_pickle=True)
    # print(a[0])
    # print(b)