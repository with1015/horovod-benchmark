import torch

class GradientHook():

    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def close(self):
        self.hook.remove()


def hook_to_model(model, backward=False):
    hooks = [GradientHook(layer[1], backward=backward) for layer in list(model._modules.items())]
    return hooks
