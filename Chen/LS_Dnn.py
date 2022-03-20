import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F

class LSDNN(nn.Module):
    def __init__(self, in_features=300, channels=1, out_features=1):
        super(LSDNN, self).__init__()

        self.f_l1 = nn.Linear(in_features, 256)
        self.f_dropout1 = nn.Dropout(p=0.5)
        self.f_relu1 = nn.ReLU()

        self.f_cnn1 = nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=3, padding=1)
        self.f_bn1 = nn.BatchNorm1d(16)
        self.f_relu2 = nn.ReLU()

        self.f_cnn2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.f_bn2 = nn.BatchNorm1d(16)
        self.f_relu3 = nn.ReLU()

        self.f_cnn3 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.f_bn3 = nn.BatchNorm1d(64)
        self.f_relu4 = nn.ReLU()

        self.f_cnn4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.f_bn4 = nn.BatchNorm1d(64)
        self.f_relu5 = nn.ReLU()

        self.id_l1 = nn.Linear(1,8)
        self.id_dropout1 = nn.Dropout(p=0.5)
        self.id_relu1 = nn.ReLU()

        self.id_l2 = nn.Linear(8,64)
        self.id_dropout2 = nn.Dropout(p=0.5)
        self.id_relu2 = nn.ReLU()

        self.id_l3 = nn.Linear(64,64)
        self.id_dropout3 = nn.Dropout(p=0.5)
        self.id_relu3 = nn.ReLU()
        self.id_bn = nn.BatchNorm1d(64)

        self.all_l1 = nn.Linear(16448, 512)
        self.all_relu1 = nn.ReLU()
        self.lstm1 = nn.LSTM(512, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.all_l2 = nn.Linear(64, out_features)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40, 60],
                                                              gamma=0.5)

    def forward(self, feature, id):

        id = self.id_l1(id)
        id = self.id_dropout1(id)
        id = self.id_relu1(id)
        id = self.id_l2(id)
        id = self.id_dropout2(id)
        id = self.id_relu2(id)
        id = self.id_l3(id)
        id = self.id_dropout3(id)
        id = self.id_relu3(id)
        id = self.id_bn(id)

        feature = self.f_l1(feature)
        feature = self.f_dropout1(feature)
        feature = self.f_relu1(feature)
        feature = feature.view(feature.size(0), 1, -1)

        feature = self.f_cnn1(feature)
        feature = self.f_bn1(feature)
        feature = self.f_relu2(feature)
        feature = self.f_cnn2(feature)
        feature = self.f_bn2(feature)
        feature = self.f_relu3(feature)
        feature = self.f_cnn3(feature)
        feature = self.f_bn3(feature)
        feature = self.f_relu4(feature)
        feature = self.f_cnn4(feature)
        feature = self.f_bn4(feature)
        feature = self.f_relu5(feature)
        feature = feature.view(feature.size(0), -1)

        concat = torch.cat((feature, id), dim=1)
        concat = self.all_l1(concat)
        concat = self.all_relu1(concat)
        concat = concat.view(concat.size(0), 1, -1)
        concat, (hn, cn) = self.lstm1(concat)
        concat, (hn, cn) = self.lstm2(concat, (hn, cn))
        concat = concat.view(concat.size(0), -1)
        out = self.all_l2(concat)

        return out




