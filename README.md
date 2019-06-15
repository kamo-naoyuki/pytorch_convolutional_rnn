# pytorch_convolutional_rnn

The pytorch implemenation for convolutional rnn is alreaedy exisitng other than my module, for example.

- https://github.com/ndrplz/ConvLSTM_pytorch
- https://github.com/jacobkimmel/pytorch_convgru

However, there are no modules supporting neither variable length tensor nor bidirectional rnn.

I implemented ``AutogradConvRNN`` by referring to ``AutogradRNN`` at https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py, so my convolutional RNN modules have similar structure to ``torch.nn.RNN`` and supports the above features as it has.

The benefit of using ``AutogradConvRNN`` is not only that it enables my modules to have the same interface as ``torch.nn.RNN``, but makes it very easy to implement many kinds of CRNN, such as ``CLSTM``, ``CGRU``.

## Require
- python3 (Not supporting python2 because I prefer type annotation)
- pytorch0.4.0, python1.0.0

## Feature
- Implemented at python level, without any additional CUDA kernel, c++ codes.
- Convolutional RNN, Convolutional LSTM, Convolutional Peephole LSTM, Convolutional GRU
- Unidirectional, Bidirectional
- 1d, 2d, 3d
- Supporting PackedSequence (Supporting variable length tensor)
- Supporting nlayers RNN and RNN Cell, both.
- Not supporting different hidden sizes for each layers (But, it is very easy to implement it by stacking 1-layer-CRNNs)

## Example
- With `pack_padded_sequence`
```python
import torch
import convolutional_rnn
from torch.nn.utils.rnn import pack_padded_sequence

in_channels = 2
net = convolutional_rnn.Conv3dGRU(in_channels=in_channels,  # Corresponds to input size
                                  out_channels=5,  # Corresponds to hidden size
                                  kernel_size=(3, 4, 6),  # Int or List[int]
                                  num_layers=2,
                                  bidirectional=True,
                                  dilation=2, stride=2, dropout=0.5)
length = 3
batchsize = 2
lengths = [3, 1]
shape = (10, 14, 18)
x = pack_padded_sequence(torch.randn(length, batchsize, in_channels, *shape), lengths, batch_first=False)
h = None
y, h = net(x, h)
```

- Without `pack_padded_sequence`
```python
import torch
import convolutional_rnn
from torch.nn.utils.rnn import pack_padded_sequence

in_channels = 2
net = convolutional_rnn.Conv2dLSTM(in_channels=in_channels,  # Corresponds to input size
                                   out_channels=5,  # Corresponds to hidden size
                                   kernel_size=3,  # Int or List[int]
                                   num_layers=2,
                                   bidirectional=True,
                                   dilation=2, stride=2, dropout=0.5,
                                   batch_first=True)
length = 3
batchsize = 2
shape = (10, 14)
x = torch.randn(batchsize, length, in_channels, *shape)
h = None
y, h = net(x, h)
```

- With `Cell`
```python
import torch
import convolutional_rnn
cell = convolutional_rnn.Conv2dLSTMCell(in_channels=3, out_channels=5, kernel_size=3).cuda()
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
