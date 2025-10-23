from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []

    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
        xs = list(map(lambda x: x, range(len(ys))))

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / 
         ((mean(xs) ** 2) - (mean(xs ** 2))))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coef_of_det(ys_orig, ys_line):
    y_mean_line = list(map(lambda y: mean(ys_orig), ys_orig))
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_y_mean)

xs, ys =  create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)
print(f"slope:{m}, intercept: {b}")

regression_line = list(map(lambda x: (m*x)+b, xs))

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coef_of_det(ys, regression_line)

print(f"R_squared:{r_squared}")
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, c='green', s=100)
plt.grid(True)
plt.show()
