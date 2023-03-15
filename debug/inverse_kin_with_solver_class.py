import time
import numpy as np
import pinocchio as pin
import hppfcl
import copy
import matplotlib.pyplot as plt


import example_robot_data as robex
from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
# Creation of the robot

robot_wrapper = RobotWrapper()
robot, rmodel, gmodel = robot_wrapper(target=True)
rdata = rmodel.createData()
gdata = gmodel.createData()

TargetID = rmodel.getFrameId('target')
assert (TargetID < len(rmodel.frames))

EndeffID = rmodel.getFrameId('endeff')
assert (EndeffID < len(rmodel.frames))

# Open the viewer
vis = create_visualizer(robot)

# Initial configuration
pin.seed(0)
q0 = pin.randomConfiguration(rmodel)
robot.q0 = q0

vis.display(q0)

target = rdata.oMf[TargetID].translation

# Optim


def residual(q):
    '''Compute score from a configuration'''
    pin.framesForwardKinematics(rmodel, rdata, q)
    p = rdata.oMf[EndeffID].translation
    return (p - target)


def cost(q):
    return 0.5 * np.linalg.norm(residual(q))


def jacobian(q):
    pin.computeJointJacobians(rmodel, rdata)
    J = pin.getFrameJacobian(
        rmodel, rdata, EndeffID, pin.LOCAL_WORLD_ALIGNED)[:3]
    return J


def gradient_cost(q):
    return np.dot(jacobian(q).T, residual(q))


def callback(q):
    vis.display(q)
    time.sleep(2e-2)

# Num diff


def numdiff(f, x, eps=1e-6):
    '''Estimate df/dx at x with finite diff of step eps'''
    xc = x.copy()
    f0 = copy.copy(f(x))
    res = []
    for i in range(len(x)):
        xc[i] += eps
        res.append(copy.copy(f(xc)-f0)/eps)
        xc[i] = x[i]
    return np.array(res).T


q = q0
Jd = numdiff(residual, q)
J = jacobian(q)
assert (np.linalg.norm(J-Jd) < 1e-5)

# Solve

pin.seed(0)

ALPHA = .1
MAX_ITER = 50
res_list = []
q_list = [q0]
cost_list = []
for i in range(MAX_ITER):
    input()
    res = residual(q)
    J = jacobian(q)
    costval = cost(q)
    q -= ALPHA * np.linalg.pinv(J)@res
    # q -= ALPHA * gradient_cost(q)
    res_list.append(np.linalg.norm(res))
    q_list.append(q)
    cost_list.append(costval)
    print(f" || {i} | {costval} ||")
    callback(q)

print('Residual at convergence : ', residual(q))
plt.plot(cost_list)
plt.show()
