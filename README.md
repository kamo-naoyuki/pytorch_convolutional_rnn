# pytorch_convolutional_rnn
## Require
- pytorch>=0.4
## Feature
- RNN, LSTM, GRU
- 1d, 2d, 3d

## Example
```python
import torch
import convolutional_rnn
from torch.nn.utils.rnn import pack_padded_sequence

net = convolutional_rnn.Conv3dGRU(2, 5, (3, 4, 6), num_layers=3, bidirectional=True,
                                  dilation=2, stride=2, dropout=0.5)
x = pack_padded_sequence(torch.randn(3, 2, 2, 10, 14, 18), [3, 1])
print(net)
y, h = net(x)
print(y.data.shape)
```
