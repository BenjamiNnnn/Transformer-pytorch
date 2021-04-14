import matplotlib.pyplot as plt
import torch
from pyitcast.transformer_utils import get_std_opt,LabelSmoothing,SimpleLossCompute
from CreateLayer import make_model
from torch.autograd import Variable
V = 5
model = make_model(V,V,N=2)

model_optimizer = get_std_opt(model)

criterion = LabelSmoothing(size=V,padding_idx=0,smoothing=0)

loss = SimpleLossCompute(model.generator,criterion,model_optimizer)

predict = Variable(torch.LongTensor([[0,0.2,0.7,0.1,0],
                                     [0,0.2,0.7,0.1,0],
                                     [0,0.2,0.7,0.1,0]]))

target = Variable(torch.LongTensor([2,1,0]))

criterion(predict,target)

plt.imshow(criterion.true_dist)