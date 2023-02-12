## More From NumPy ##

import numpy as np

np.random.seed(123)
a1 = np.array([3, 8, 11, 22, 7])
a2 = np.random.randint(0, 50, size=(3, 5))

## Comparison
print('\n(a1 == a2): \n', (a1 == a2))
print('\n(a1 < a2): \n', (a1 < a2))
print('\n(a1 >= a2): \n', (a1 >= a2))

bool_array = (a1 >= a2)
print('\nbool_array: \n', bool_array)
print('\nbool_array type: ', type(bool_array))
print('\nbool_array datatype: ', bool_array.dtype)

## Sorting
print('\na2: \n', a2)
print('\nnp.sort(a2): \n', np.sort(a2))
print('\nnp.argsort(a2): \n', np.argsort(a2))
print('\nnp.argmin(a2): \n', np.argmin(a2))
print('\nnp.argmax(a2): \n', np.argmax(a2))
print('\nnp.argmax(a2, axis=0): \n', np.argmax(a2, axis=0))
print('\nnp.argmax(a2, axis=1): \n', np.argmax(a2, axis=1))

## Turn image to NumPy arrays
from matplotlib.image import imread

panda = imread('../data/numpy-images/panda.png')
print('\npanda data: \n', panda[:2])
print('\npanda information: \ntype: {0} \ndatatype: {1} \nsize: {2} \nshape: {3} \ndimension: {4}'
      .format(type(panda), panda.dtype, panda.size, panda.shape, panda.ndim))

dog = imread('../data/numpy-images/dog-photo.png')
print('\ndog top 3 data: \n', dog[:3])
print('\ndog information: \ntype: {0} \ndatatype: {1} \nsize: {2} \nshape: {3} \ndimension: {4}'
      .format(type(dog), dog.dtype, dog.size, dog.shape, dog.ndim))

car = imread('../data/numpy-images/car-photo.png')
print('\ncar top 3 data: \n', car[:3])
print('\ncar information: \ntype: {0} \ndatatype: {1} \nsize: {2} \nshape: {3} \ndimension: {4}'
      .format(type(car), car.dtype, car.size, car.shape, car.ndim))
