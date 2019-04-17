from collections import OrderedDict
import torch
import torch.nn as nn
import os
import hickle
from models.normalize import Normalize
import torch.nn.functional as F

# originate from
# https://github.com/vadimkantorov/metriclearningbench

class inception_v1_googlenet(nn.Sequential):
	output_size = 1024
	input_side = 227
	rescale = 255.0
	rgb_mean = [122.7717, 115.9465, 102.9801]
	rgb_std = [1, 1, 1]

	def __init__(self):
		super(inception_v1_googlenet, self).__init__(OrderedDict([
			('conv1', nn.Sequential(OrderedDict([
				('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))),
				('relu1', nn.ReLU(True)),
				('pool1', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),
				('lrn1', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1))
			]))),

			('conv2', nn.Sequential(OrderedDict([
				('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))),
				('relu1', nn.ReLU(True)),
				('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1))),
				('relu2', nn.ReLU(True)),
				('lrn2', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)),
				('pool2', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True))
			]))),

			('inception_3a', InceptionModule(192, 64, 96, 128, 16, 32, 32)),
			('inception_3b', InceptionModule(256, 128, 128, 192, 32, 96, 64)),

			('pool3', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),

			('inception_4a', InceptionModule(480, 192, 96, 208, 16, 48, 64)),
			('inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)),
			('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64, 64)),
			('inception_4d', InceptionModule(512, 112, 144, 288, 32, 64, 64)),
			('inception_4e', InceptionModule(528, 256, 160, 320, 32, 128, 128)),

			('pool4', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),

			('inception_5a', InceptionModule(832, 256, 160, 320, 32, 128, 128)),
			('inception_5b', InceptionModule(832, 384, 192, 384, 48, 128, 128)),

			('pool5', nn.AvgPool2d((7, 7), (1, 1), ceil_mode = True)),

			#('drop5', nn.Dropout(0.4))
		]))

class InceptionModule(nn.Module):
	def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5, outplane_pool_proj):
		super(InceptionModule, self).__init__()
		a = nn.Sequential(OrderedDict([
			('1x1', nn.Conv2d(inplane, outplane_a1x1, (1, 1), (1, 1), (0, 0))),
			('1x1_relu', nn.ReLU(True))
		]))

		b = nn.Sequential(OrderedDict([
			('3x3_reduce', nn.Conv2d(inplane, outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0))),
			('3x3_relu1', nn.ReLU(True)),
			('3x3', nn.Conv2d(outplane_b3x3_reduce, outplane_b3x3, (3, 3), (1, 1), (1, 1))),
			('3x3_relu2', nn.ReLU(True))
		]))

		c = nn.Sequential(OrderedDict([
			('5x5_reduce', nn.Conv2d(inplane, outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0))),
			('5x5_relu1', nn.ReLU(True)),
			('5x5', nn.Conv2d(outplane_c5x5_reduce, outplane_c5x5, (5, 5), (1, 1), (2, 2))),
			('5x5_relu2', nn.ReLU(True))
		]))

		d = nn.Sequential(OrderedDict([
			('pool_pool', nn.MaxPool2d((3, 3), (1, 1), (1, 1))),
			('pool_proj', nn.Conv2d(inplane, outplane_pool_proj, (1, 1), (1, 1), (0, 0))),
			('pool_relu', nn.ReLU(True))
		]))
		
		for container in [a, b, c, d]:
			for name, module in container.named_children():
				self.add_module(name, module)

		self.branches = [a, b, c, d]

	def forward(self, input):
		return torch.cat([branch(input) for branch in self.branches], 1)
      

class Model(nn.Module):
    def __init__(self, base_model, low_dim = 128):
        super(Model, self).__init__()
        self.base_model = base_model
        self.embedder = nn.Linear(base_model.output_size, low_dim)
        self.l2norm = Normalize(2)
        
        # base_model = inception_v1_googlenet()
    def forward(self, input):
        pool5 = self.base_model(input).view(len(input), -1)
        embed = self.embedder(pool5)
        embed = self.l2norm(embed)
        if self.training:
            return embed
        else:
            return embed, self.l2norm(pool5)
        
        
def inception_v1_ml(pretrained=False,low_dim=128):
    base_model = inception_v1_googlenet()
    base_model_weights_path = 'models/googlenet.h5'
    if os.path.exists(base_model_weights_path):
        base_model.load_state_dict({k : torch.from_numpy(v).cuda() for k, v in hickle.load(base_model_weights_path).items()}) 
    net = Model(base_model, low_dim)
    return net
