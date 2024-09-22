import matplotlib.pyplot as plt
import math

x = [0.0]
y = [1.0]
diff = 10.0/1000.0
for i in range(1, 1000):
    x.append(i*diff)
    y.append(math.exp(-i*diff))

plt.plot(x,y)
plt.show()