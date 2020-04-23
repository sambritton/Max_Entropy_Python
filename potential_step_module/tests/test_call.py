
import numpy as np
import pickle as P
import os

datadir = './test_cases'

inputs = P.load(open(os.path.join(datadir, 'inputs.p'), 'rb'))
expected = P.load(open(os.path.join(datadir, 'results.p'), 'rb'))

import pstep

print(f'Inputs: ')
for k, v in inputs.items():
    print(f'Shape for {k}: {v.shape}')

assert len(inputs) == 7, 'Wrong number of inputs passed to dispatch'
actual = pstep.dispatch([i for i in range(len(inputs['state']))], *inputs.values());
print('Call successful.')

epsilon = 0.0000001

for i in len(rets):
    for j in len(rets[i]):
        if abs(expected[i, j] - actual[i, j]) > epsilon:
            print(f'--- Expected<{expected[i, j]}> Actual<{actual[i, j]}>')

print(f'Got: {rets}')
