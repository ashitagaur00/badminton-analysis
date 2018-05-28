from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #x = F.log_softmax(x, dim=1)
        return F.tanh(x)

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return x1, x2

def train(args, model, device, train_loader1, train_loader2, optimizer, criterion, epoch):
    model.train()
    for batch_idx, ((data1, target1), (data2, target2)) in enumerate(zip(train_loader1, train_loader2)):
        data1 = data1.to(device)
        data2 = data2.to(device)
        target = (target1>target2).type(torch.FloatTensor) - (~(target1 > target2)).type(torch.FloatTensor)
        target = target.view(16).to(device)
        optimizer.zero_grad()
        output1, output2 = model(data1, data2)
        loss = criterion(output1, output2, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader1.dataset),
                100. * batch_idx / len(train_loader1), loss.item()))

def test(args, model, device, test_loader1, test_loader2, criterion):
    model.eval()
    #test_loss = 0
    correct = 0
    with torch.no_grad():
        for ((data1, target1), (data2, target2)) in zip(test_loader1, test_loader2):
            data1 = data1.to(device)
            data2 = data2.to(device)
            target = (target1>target2).type(torch.FloatTensor) - (~(target1 > target2)).type(torch.FloatTensor)
            target = target.view(16)
            output1, output2 = model(data1, data2)
            #test_loss += criterion(output1, output2, target, size_average=False).item() # sum up batch loss
            pred = (output1>output2).type(torch.FloatTensor) - (~(output1 > output2)).type(torch.FloatTensor)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(zip([ x[0] for x in output1.cpu().numpy().tolist()], target1.cpu().numpy().tolist()))
    print()
    #test_loss /= len(test_loader1.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader1.dataset),
        100. * correct / len(test_loader1.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader1 = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader1 = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_loader2 = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.MarginRankingLoss(margin=0.2)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader1, train_loader2, optimizer, criterion, epoch)
        test(args, model, device, test_loader1, test_loader2, criterion)


if __name__ == '__main__':
    main()
