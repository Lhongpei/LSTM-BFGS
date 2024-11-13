import torch
import torchmin.bfgs as bfs
import matplotlib.pyplot as plt
import numpy as np
# 定义一个二次函数作为例子
def quadratic_function(x):
    return 0.5 * (x+1).dot(x-1) + 3 * sum(x) + 5

# 定义二次函数的梯度
def gradient_quadratic_function(x):
    return x + 3

# 使用 BFGS 方法进行最优化
def optimize_quadratic_function():
    # 初始化优化的起始点
    x0 = torch.tensor([0.0,0.0], requires_grad=True)

    # 定义优化的目标函数
    def objective_function(x):
        return quadratic_function(x)

    # 进行 BFGS 最优化
    result = bfs._minimize_bfgs(
        fun=objective_function,
        x0=x0,
        lr=0.1,
        inv_hess=True,
        max_iter=100,
        line_search='strong-wolfe',
        gtol=1e-5,
        xtol=1e-9,
        normp=float('inf'),
        callback=None,
        disp=1,
        return_all=True
    )

    # 输出最优结果
    return(result)

# 调用函数进行优化
result=optimize_quadratic_function()
opt_val=result['fun']
x=[np.array(i) for i in result['allvecs']]
error=[quadratic_function(i)-opt_val for i in x]
plt.plot(error)