import torch
import Loader
import itertools
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lifelines.utils import concordance_index as con_index
from torch.utils.tensorboard import SummaryWriter
from Model import MSSTrans
from NegativeLogLikelihood import CoxLoss

from thop import profile
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Options -------
bch_size_train = 16
bch_size_test = 16
epoch_size = 100
base_lr = 0.001
writer = SummaryWriter('./log')
# ----------------------

transfers = transforms.Compose([
    transforms.ToTensor()
])

result = []
kfold = 1  # In practice, please set to 5 or other
for cluster in range(kfold):
    print("************** SPLIT (%d/%d) **************" % (cluster+1, kfold))
    train0 = Loader.PatchData.split_cluster('./data/gbmlgg/TCGA-GBMLGG.csv', 'Train', cluster, transfer=transfers)
    valid0 = Loader.PatchData.split_cluster('./data/gbmlgg/TCGA-GBMLGG.csv', 'Valid', cluster, transfer=transfers)
    dataloader = DataLoader(train0, batch_size=bch_size_train, shuffle=True, num_workers=0)
    dataloader_var = DataLoader(valid0, batch_size=bch_size_test, shuffle=True, num_workers=0)

    model = MSSTrans()
    model.to(device)

    inputs = torch.randn(16, 1, 64, 256).to(device)
    flops, _ = profile(model, inputs=(inputs,inputs,inputs))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)

    for epoch in range(epoch_size):
        loss_board = 0
        model.train()
        for iteration, data in enumerate(dataloader):
            img_feature_4096, img_feature_256, img_feature_16, s_label, c_label, _ = data
            img_feature_4096 = Variable(img_feature_4096, requires_grad=False).to(device)
            img_feature_256 = Variable(img_feature_256, requires_grad=False).to(device)
            img_feature_16 = Variable(img_feature_16, requires_grad=False).to(device)
            s_label = Variable(s_label, requires_grad=False).to(device)
            c_label = Variable(c_label, requires_grad=False).to(device)

            pred = model(img_feature_4096, img_feature_256, img_feature_16)

            loss = CoxLoss(s_label, pred, 'cuda')
            print('Epoch: {}/({})'.format(iteration, epoch+1), 'Train_loss: %.4f' %(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_board += loss
        writer.add_scalar("Train_loss", (loss_board-1.5), epoch)

        # Validation
        model.eval()
        c_index = 0
        accuracy = 0
        n = 0
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        for iteration, data in enumerate(dataloader_var):
            n += 1.0
            with torch.no_grad():
                img_feature_4096, img_feature_256, img_feature_16, s_label, c_label, _ = data
                img_var_4096 = Variable(img_feature_4096, requires_grad=False).to(device)
                img_var_256 = Variable(img_feature_256, requires_grad=False).to(device)
                img_var_16 = Variable(img_feature_16, requires_grad=False).to(device)
                s_label_var = Variable(s_label, requires_grad=False).to(device)
                c_label_var = Variable(c_label, requires_grad=False).to(device)
                ytime, yevent = s_label_var[:, 0], s_label_var[:, 1]

                start_time = time.time()
                pred_test = model(img_var_4096, img_var_256, img_var_16)
                end_time = time.time()
                inference_time = end_time - start_time
                # print(f"模型推理时间: {inference_time:.6f} 秒")

                pred_test = pred_test.reshape(-1)
            y, risk_pred, e = ytime.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), yevent.detach().cpu().numpy()
            # print(y, risk_pred)
            risk_pred_all = np.concatenate((risk_pred_all, risk_pred.reshape(-1)))
            censor_all = np.concatenate((censor_all, e.reshape(-1)))
            survtime_all = np.concatenate((survtime_all, y.reshape(-1)))
        try:
            c_index = con_index(survtime_all, -risk_pred_all, censor_all)
        except:
            print('No admissable pairs in the dataset.')
        print('Epoch(' + str(epoch + 1) + ')',  'Train_loss: %.4f' % (loss_board.item()), 'Test_acc: %.4f' % (c_index))
        writer.add_scalar("Test_acc", c_index, epoch)

        result.append(c_index)

        # if epoch % 10 == 9:
        #     torch.save(omics_model.state_dict(), './model/omics_model_{}.pt'.format(epoch+1))
        #     torch.save(model.state_dict(), './model/model_{}.pt'.format(epoch+1))
    print(result)
    print('max: {}, min: {}, mean c-index: {}'.format(max(result), min(result), sum(result)/len(result)))
