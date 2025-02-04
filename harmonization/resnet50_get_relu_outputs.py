import torch
import torchvision.models as models
from torch import nn

def resnet_output(model_resnet, input_img):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)



    # Dictionary to store the activations
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    # Get all ReLU layer names and register hooks
    relu_names = []
    if model_resnet._get_name()=='VGG':
        for name, module in model_resnet.features.named_modules():
            if isinstance(module, torch.nn.ReLU):
                relu_names.append(name)
                module.register_forward_hook(get_activation(name))
    else:
        for name, module in model_resnet.named_modules():
            if isinstance(module, torch.nn.ReLU):
                relu_names.append(name)
                module.register_forward_hook(get_activation(name))

    # print("ReLU layers with registered hooks:", relu_names)

    # Prepare input
    input_tensor = input_img  # Example input, adjust size if needed

    # Forward pass
    with torch.no_grad():
        output = model_resnet(input_tensor)

    # Access and print the extracted features
    # for name in relu_names:
    #     if name in activation:
    #         print(f"{name} output shape:", activation[name].shape)

    return activation, relu_names