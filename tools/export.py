import torch
from torchvision import transforms
from PIL import Image
import time
import sys 
sys.path.append(".") 
from model import transformer


if __name__ == "__main__":
    preprocess_transform = transforms.Compose([
         transforms.ToTensor(),
     ])
    device = torch.device('cuda')
    img = Image.open('./data/26.jpg')
    img_tensor = preprocess_transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    input_dict = dict(x=img_tensor)
    model = transformer.Net()
    model = torch.load('./weights/weight.pth')
    # model.load_state_dict('weight.pth')

    model.eval()
    model.to(device)

    # export onnx 
    torch.onnx.export(
        model,
        input_dict,
        "./weights/weight.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
    )

    # model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted = torch.jit.trace(model,input_dict)
    model_scripted.save('./weights/model_scripted.pt') # Save 
    
    for _ in range(10):
        start = time.time()
        out = model(input_dict)
        pred = out.data.max(1, keepdim=True)[1]
        print(f"runtime :{time.time() - start}s")
    print(out.data)
    print("-----------------.pth result",pred)

    import onnx

    onnx_model = onnx.load("./weights/weight.onnx")
    print(onnx.checker.check_model(onnx_model))


    # export torch script model

    
    
