# Getting started

We use a functional API and try to follow most of PyTorch's convention.
Check out the API and start playing.

All API functions live under `nitorch_interpol` and are imported by default.
_E.g._
```python
from nitorch_interpol import pull
resampled = pull(signal, coordinates, order=3, prefilter=True)
```
or
```python
import nitorch_interpol as interpol
resampled = interpol.pull(signal, coordinates, order=3, prefilter=True)
```

Notable differences include the way sampling coordinates are encoded 
(`[0, N-1]` instead of `[-1, 1]`, no index flipping), and the fact that 
we often use a "channel last" dimension ordering.
