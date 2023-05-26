from engines.engine import BaseEngine
from torch.nn import Module
import torch

class DNNEngine(BaseEngine):
    def __init__(self,model_path,device):
        super(DNNEngine, self).__init__("dnnengine")
        self.model = torch.load(model_path).to(device)
        self.device = device
        self.output = None

    def update_engine_input(self,input):
        self.input = input
        self.input = torch.Tensor(self.input).to(self.device)



    def execute_engine(self,debug=False):
        self.output = self.model.forward(self.input)

    def get_output(self):
        return self.output.detach().cpu().numpy().tolist()


