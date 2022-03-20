import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from dataloader import DataLoader
from LS_Dnn import LSDNN

device = torch.device("cpu")

Dataset = DataLoader(batch_size=1024)
Dataset.initialize_data()

n_iter = Dataset.batch_num

model = LSDNN().to(device)

criterion = model.criterion
optimizer = model.optimizer
scheduler = model.scheduler

def train(epoch):
    print('===> Training # %d epoch' % (epoch))
    epoch_loss = 0
    model.train()
    for i in range(n_iter):
        investment_id, feature, target = Dataset.get_batch_data()
        investment_id, feature, target = investment_id.to(device), feature.to(device), target.to(device)

        optimizer.zero_grad()
        model_out = model(feature, investment_id)

        lossl2 = criterion(model_out, target)
        epoch_loss += lossl2.item()

        lossl2.backward()

        optimizer.step()

        print("===> Epoch[{}]({}/{}): Train Loss: {:.6f}".format(epoch, i, n_iter,
                                                                 lossl2.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / n_iter))

def checkpoint(epoch):
    model_out_path = './LSDNN_epoch_{}'.format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    nEpochs = 100
    for epoch in range( nEpochs ):
        print("=====>  Training %d epochs"%(epoch))
        train(epoch)
        print("=====>  Training %d epochs completed"%(epoch))
        scheduler.step()
        print("=====>  lr scheduler activated in %d epochs completed" % (epoch))
        if epoch % 10 == 0:
            # if listpsnr[-1] == np.max(listpsnr) or epoch == nEpochs:
            print("=====>  Save checkpoint %d epochs" % (epoch))
            #            print("=====>  best D_PSNR %f"%(listpsnr[-1]))
            checkpoint(epoch)
            print("=====>  Save checkpoint %d epochs completed" % (epoch))
