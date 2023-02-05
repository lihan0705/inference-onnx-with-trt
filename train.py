import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from model.transformer import Net
import torch.optim as optim
import time

n_epochs = 1
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



if __name__ == "__main__":
  network = Net()
  optimizer = optim.SGD(network.parameters(), lr=0.01,
                        momentum=0.5)

  DOWNLOAD_PATH = 'data/mnist'
  BATCH_SIZE_TRAIN = 100
  BATCH_SIZE_TEST = 100
  device = torch.device('cuda')


  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),batch_size=batch_size_test, shuffle=True)

  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  def train(epoch):
    network.train()
    network.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      data = data.to(device)
      target = target.to(device)
      start = time.time()
      input_dict = dict(x=data)
      output = network(input_dict)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\t every batch cost time : {:.6f} '.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(), time.time() - start))

        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        #torch.save(network.state_dict(), 'results/model.pth')
        #torch.save(optimizer.state_dict(), 'results/optimizer.pth')
      torch.save(network, 'weights/weight.pth')
      torch.save(network.state_dict(), 'weights/model_state_dict.pth')
      #model_scripted = torch.jit.script(network) # Export to TorchScript
      #model_scripted = torch.jit.trace(network,input_dict)
      #model_scripted.save('model_scripted.pt') # Save 
      

  def test():
    network.eval()
    network.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        input_dict = dict(x=data)
        output = network(input_dict)
        output = network(input_dict)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

  for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()