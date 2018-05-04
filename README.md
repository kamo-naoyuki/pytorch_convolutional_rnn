# pytorch_convolutional_rnn
## Require
- python3 (Not supporting python2 because I prefer type annotation)
- pytorch>=0.4

## Feature
- Autograd version
- Convolutional RNN, Convolutional LSTM, Convolutional Peephole LSTM, Convolutional GRU
- Unidirectional, Bidirectional
- 1d, 2d, 3d
- Supporting PackedSequence
- RNN Cell

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


cell = convolutional_rnn.Conv2dLSTMCell(3, 5, 3).cuda()
time = 6
input = torch.randn(time, 16, 3, 10, 10).cuda()
output = []
for i in range(time):
    if i == 0:
        hx, cx = cell(input[i])
    else:
        hx, cx = cell(input[i], (hx, cx))
    output.append(hx)

```
