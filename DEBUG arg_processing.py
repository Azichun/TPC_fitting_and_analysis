from nlsfunc import *

# should return [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
line(np.linspace(1, 10, 10), 1, 2)
line(np.linspace(1, 10, 10), [1, 2])
line(np.linspace(1, 10, 10), {"m": 1, "c": 2})
line(np.linspace(1, 10, 10), m=1, c=2)

# should return 3
line(1, m=1, c=2)

# should raise warning
line(np.linspace(1, 10, 10), [3, 4], m=1, c=2)

# should raise error
line(np.linspace(1, 10, 10), [1])
line(np.linspace(1, 10, 10), [1, 2, 3])
line(np.linspace(1, 10, 10), 1, 2, 3)
line(np.linspace(1, 10, 10), 1)
line(np.linspace(1, 10, 10), m=1, c=2, k=3)