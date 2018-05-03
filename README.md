# pytorch_convolutional_rnn
## Require
- python3 (Not supporting python2 because I prefer type annotation)
- pytorch>=0.4

## Feature
- Autograd version
- RNN, LSTM, Peephole LSTM, GRU
- Unidirectional, Bidirectional
- 1d, 2d, 3d
- Supporting PackedSequence

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
