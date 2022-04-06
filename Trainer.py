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
        """
        初始化模型，优化器，损失函数
        :return:
        """
        self.net = DCISCF()
        self.net.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        self.cri = nn.CrossEntropyLoss()

    def _init_data(self):
        """
        初始化数据
        :return:
        """
        data = DL(self.args)

        self.traindl = data.traindl
        self.testdl = data.testdl

    def save_model(self):
        """
        保存模型
        :return:
        """
        state_dict = self.net.state_dict()
        torch.save(state_dict, self.args.results_path + 'net.pt')

    def load_model(self):
        """
        加载模型
        :return:
        """
        self.net.load_state_dict(torch.load(self.args.results_path + 'net.pt', map_location='cpu'))
        self.net.to(self.device)

    def train(self):
        """
        训练网络
        :return:
        """
        patten = 'epochs: %d/%d   [================]   loss:%.4f    acc:%.5f     eval_acc: %.5f'
        for epoch in range(1, self.args.epochs + 1):
            pred = []
            real = []
            losses = 0
            for batch, (inputs, targets) in enumerate(self.traindl):
                inputs = inputs.to(self.device)          # 数据转移到gpu上
                targets = targets.to(self.device).squeeze()
                n = inputs.shape[0]
                ml = list(range(n))
                mask = torch.zeros(n, 3)        # 设置label的掩码
                mask[ml, targets] = 1
                mask -= 0.3
                mask = F.normalize(mask.abs(), p=1, dim=1).to(self.device)   # ，做标签平滑处理，并做L1归一化
                output = self.net(inputs)
                pred = pred + output.argmax(dim=1).squeeze().detach().cpu().numpy().tolist()
                real = real + targets.detach().cpu().numpy().tolist()
                loss = self.cri(output, mask)
                self.opt.zero_grad()       # 梯度清零
                loss.backward()        # 反向传播
                self.opt.step()        # step

                losses += loss.item()
                torch.cuda.empty_cache()
            pred = torch.tensor(pred)
            real = torch.tensor(real)
            acc = (pred == real).sum() / pred.shape[0]

            eval_acc = self.eval()
            print(patten % (
                epoch,
                self.args.epochs,
                losses,
                acc,
                eval_acc,
            ))
        self.save_model()

    def eval(self):
        """
        验证
        :return:
        """
        # self.load_model()
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
            torch.cuda.empty_cache()
        pred = torch.tensor(preds)
        real = torch.tensor(reals)
        acc = (pred == real).sum() / pred.shape[0]
        self.net.train()
        return acc
