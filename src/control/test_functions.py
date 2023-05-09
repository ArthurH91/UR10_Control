# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from solver_base import Solver

def test_function_1(x):
    f = np.sum(np.square(x))
    return f

def test_gradient_function_1(x):
    gf = 2 * x
    return gf

def test_function_rosenbrock(x):
    a = 1
    b = 100
    f = np.sum(np.square(x[::2] - a) + b * np.square(np.square(x[::2]) - x[1::2]))
    return f

def test_gradient_function_rosenbrock(x):
    a = 1 
    b = 100
    gf = np.zeros(x.shape)
    gf[::2] = 2 * (x[::2] - a) + 4 * b * (np.square(x[::2]) - x[1::2]) * x[::2]
    gf[1::2] = -2 * b * (np.square(x[::2]) - x[1::2])
    return gf


def compute_rosenbrock_function(X: np.ndarray, a: float = 1., b: float = 100.):
    "Returns the value of the banana of rosenbrock given a x, y and a and b parameters"
    x, y = X
    return (a - x)**2 + b * (y - x ** 2)**2


def compute_grad_rosenbrock_function(X: np.ndarray, a: float = 1., b: float = 100.):
    "Returns the value of the gradient of the banana of rosenbrock given a x,y and 2 parameters a and b"
    x, y = X
    return np.array([2 * (x-a) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])


def compute_hess_ros_function(X: np.ndarray, a: float = 1, b: float = 100.):
    "Returns the hessian value of the rosenbrock function given a x, y and 2 parameters a and b"
    x, y = X
    H = np.eye(2)
    H[0, 0] = 2 - 4 * b * (y - 3 * x ** 2)
    H[0, 1] = - 4 * b * x
    H[1, 0] = - 4 * b * x
    H[1, 1] = 2 * b
    return H

if __name__ == "__main__":
    x0 = np.array([3, 4])
    alpha = 1
    # print(line_search(test_function_1, test_gradient_function_1, alpha, x0, eps=1e-6))

    # print("Testing function Rosenbrock:")
    # x0 = np.array([3, 4, 0, 0])
    # alpha = 1
    # print(line_search(test_function_rosenbrock, test_gradient_function_rosenbrock, alpha, x0, eps=1e-6, max_iter=10000))

    backtracking_linesearch = Solver(compute_rosenbrock_function, compute_grad_rosenbrock_function,compute_hess_ros_function, alpha = alpha, max_iter= 10000, step_type="newton")
    backtracking_linesearch(x0)