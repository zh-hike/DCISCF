import argparse
from Trainer import Trainer



"""
设置网络参数
"""
parse = argparse.ArgumentParser('DCISCF')

parse.add_argument('--data_path', type=str, help="数据集路径", default="./dataset/data/")
parse.add_argument('--results_path', type=str, help="结果保存路径", default='./results/')
parse.add_argument('--batch_size', type=int, help="批次大小", default=50)
parse.add_argument('--epochs', type=int, help="轮次", default=50)
parse.add_argument('--device', type=int, choices=[-1, 0, 1, 2, 3], help="训练位置，-1代表cpu", default=0)
parse.add_argument('--size', type=int, nargs='+', help="W和H", default=[256, 256])
parse.add_argument('--train', action='store_true')
parse.add_argument('--eval', action='store_true')
parse.add_argument('--time', type=int, help="验证集切片次数", default=1)
args = parse.parse_args()

if __name__ == "__main__":
    trainer = Trainer(args)

    if args.train:
        trainer.train()
    if args.eval:
        trainer.eval()
