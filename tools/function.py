# from torch import package
import torch
import argparse
"""
path = '/home/liangdao_hanli/Desktop/task/transformer/model_scripted.pt'

package_name = 'mnist_inferece'
resource_name = "model.pkl"
# model = torch.load(path)

imp = package.PackageImporter(path)
loaded_model = imp.load_pickle(package_name, resource_name)

print(loaded_model)
"""
def build_net():
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument('--cfg_file', type=str, required = True, help='specify the config for training')
    args = parser.parse_args()
    model = torch.jit.load(args.cfg_file)
    print(model)

build_net()
