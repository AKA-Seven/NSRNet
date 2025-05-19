
from SRM import OperationAttn,TaskAdaptor
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# NAFBlock = NAFBlock(c=9)
# print(count_parameters(NAFBlock))
OperationAttn = OperationAttn(3,2,3)
task_adaptor = TaskAdaptor(in_size=16 * 2**3,out_size=3,semodule= None)


print(count_parameters(OperationAttn))
print(count_parameters(task_adaptor))    
    
