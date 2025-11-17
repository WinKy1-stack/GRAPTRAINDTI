import pickle
import numpy as np

h = pickle.load(open('./checkpoints/GraphTransDTI_KIBA_history.pkl', 'rb'))
vr = h['val_rmse']

print('Total length:', len(vr))
print('Full array min:', min(vr))
print('Argmin:', np.argmin(vr))
print('Value at best:', vr[np.argmin(vr)])
print('\nAround best index:')
best_idx = np.argmin(vr)
for i in range(max(0, best_idx-3), min(len(vr), best_idx+4)):
    print(f'  Index {i}: {vr[i]:.6f}')

print('\nIf array has 2x entries (train+val per epoch):')
print(f'  Best index {best_idx} -> Epoch {best_idx//2 + 1}')

print('\nFirst 10 entries:', vr[:10])
