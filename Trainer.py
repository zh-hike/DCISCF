from dataset.dataLoader import DL
import torch
import torch.nn as nn
from network import DCISCF
import torch.nn.functional as F


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        if self.args.device != -1:
            self.device = torch.device('cuda:%d' % self.args.device)

        self._init_model()
        self._init_data()

    def _init_model(self):
        self.net = DCISCF()
        self.net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        self.cri = nn.CrossEntropyLoss()

    def _init_data(self):
        data = DL(self.args)

        self.traindl = data.traindl
        self.testdl = data.testdl

    def save_model(self):
        state_dict = self.net.state_dict()
        torch.save(state_dict, self.args.results_path + 'net.pt')

    def load_model(self):
        self.net.load_state_dict(torch.load(self.args.results_path + 'net.pt', map_location='cpu'))
        self.net.to(self.device)

    def train(self):
        patten = 'epochs: %d/%d   [================]   loss:%.4f    acc:%.5f'
        for epoch in range(1, self.args.epochs + 1):
            pred = []
            real = []
            losses = 0
            for batch, (inputs, targets) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).squeeze()

                output = self.net(inputs)
                pred = pred + output.argmax(dim=1).squeeze().detach().cpu().numpy().tolist()
                real = real + targets.detach().cpu().numpy().tolist()
                loss = self.cri(output, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                losses += loss.item()

            pred = torch.tensor(pred)
            real = torch.tensor(real)
            acc = (pred == real).sum() / pred.shape[0]
            print(patten % (
                epoch,
                self.args.epochs,
                losses,
                acc,
            ))
        self.save_model()

    def eval(self):
        self.load_model()
        self.net.eval()
        preds = []
        reals = []
        for batch, (inputs, targets) in enumerate(self.testdl, 1):
            n = targets.shape[0]
            pred = 0
            targets = targets.to(self.device)
            time = len(inputs)
            for x in inputs:
                x = x.to(self.device)
                output = self.net(x)
                output = F.softmax(output, dim=1)
                pred += output
            pred /= time
            preds = preds + pred.argmax(dim=1).squeeze().detach().cpu().numpy().tolist()
            reals = reals + targets.detach().cpu().numpy().tolist()

        pred = torch.tensor(preds)
        real = torch.tensor(reals)
        acc = (pred == real).sum() / pred.shape[0]
        print("验证acc: %.5f" % acc)
