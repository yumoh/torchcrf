
from torchcrf import CRF
import torch
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_length, batch_size, num_tags = 3, 2, 5
emissions = torch.randn(seq_length, batch_size, num_tags,requires_grad=True)
tags = torch.LongTensor([[0, 1], [2, 4], [3, 1]])
emissions=emissions.to(device)
tags=tags.to(device)
model = CRF(num_tags).to(device)

print(model(emissions, tags))

print(model.decode(emissions))

mask = torch.ByteTensor([[1, 1], [1, 1], [1, 0]])
mask=mask.to(device)
r=model(emissions, tags, mask=mask)
print(r)