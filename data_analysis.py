import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats


def q1():
    """
    1、使用均匀分布获得指数分布，只需要先获取均匀分布，再求解指数的反函数，对数函数的值即可。
    """
    uniform_a = np.random.uniform(0, 1, 2000)
    exponential_b = [-math.log(i) for i in uniform_a]
    pillar = 50
    x = np.linspace(0, 10, 1000)
    p = stats.expon.pdf(x, loc=0, scale=1)
    plt.plot(x, p, label='pdf', color='k')
    a = plt.hist(exponential_b, pillar, color='g', density=1)
    plt.plot(a[1][0:pillar], a[0], 'r')
    plt.grid()
    plt.show()


def q2():
    """
    x = x1 + x2 + x3 + x4 + x5
    y = -3e**-3x
    :return:
    """
    exponential_b = []
    for i in range(5):
        uniform_a = np.random.uniform(0, 1, 2000)
        exponential_b += [-math.log(i) for i in uniform_a]
    pillar = 50
    x = np.linspace(0, 10, 1000)
    p = stats.expon.pdf(x, loc=0, scale=1)
    plt.plot(x, p, label='pdf', color='k')
    a = plt.hist(exponential_b, pillar, color='g', density=1)
    plt.plot(a[1][0:pillar], a[0], 'r')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    q2()











































