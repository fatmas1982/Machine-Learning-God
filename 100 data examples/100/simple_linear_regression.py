import numpy as np
import matplotlib.pyplot as plt

# generate training data
x = np.arange(0, 10, 0.1)
y = 1 + (x * 2) + (np.random.normal(0, 1, len(x)) * 5)

# compute coefficients for simple linear regression
mx = x.mean()
my = y.mean()
temp = x - mx
c1 = np.sum(temp * (y - my)) / np.sum(temp ** 2)
c0 = my - c1 * mx

# plot -----------------------------------------------------

x2 = [0,10]
y2 = [c0 + c1*0, c0 + c1*10]

my_dpi = 96
plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

plt.scatter(x,y, color='b', s=20)
plt.plot(x2,y2, color='r', linewidth=3)

plt.axis([0,10,-5,30])

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression\n')

plt.savefig("simple_lin_regression.png")

# plot -----------------------------------------------------