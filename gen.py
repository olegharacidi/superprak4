import random

N = 100
print N
for i in xrange(N):
  s = "%03d " * N
  v = tuple(random.randrange(1, 1000) for t in xrange(N));
  print s % v
