
import numpy as np

import pstep

print('Calling dispatch with List[int] and Dict[str, Any]')
pstep.dispatch(
        [ i for i in range(10) ],
        [np.ones(10, dtype=float),
         np.ones(10, dtype=float),
         np.ones(10, dtype=float),
         np.ones(10, dtype=float),
         np.ones(10, dtype=float),
         np.ones(10, dtype=float),
         np.ones(10, dtype=float),
         np.ones(10, dtype=float)]);
print('Call successful.')
