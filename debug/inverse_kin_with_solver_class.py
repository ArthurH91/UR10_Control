import time
import numpy as np
import pinocchio as pin
import hppfcl
import copy
import matplotlib.pyplot as plt


import example_robot_data as robex
from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from Solver import SolverWithDisplay

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

# Optimization functions


def residual(q: np.ndarray):
    """Compute residuals from a configuration q. 
    Here, the residuals are calculated by the difference between the cartesian position of the end effector and the target.

    Parameters
    ----------
    q : np.ndarray
        Array of configuration of the robot, size rmodel.nq.

    Returns
    -------
    residual : np.ndarray
        Array of the residuals at a configuration q, size 3. 
    """

    # Forward kinematics of the robot at the configuration q.
    pin.framesForwardKinematics(rmodel, rdata, q)

    # Obtaining the cartesian position of the end effector.
    p = rdata.oMf[EndeffID].translation
    return (p - target)


def cost(q : np.ndarray):
    """Compute the cost of the configuration q. The cost is quadratic here.

    Parameters
    ----------
    q : np.ndarray
        Array of configuration of the robot, size rmodel.nq.

    Returns
    -------
    cost : float
        Cost of the configuration q. 
    """
    return 0.5 * np.linalg.norm(residual(q))


def jacobian(q : np.ndarray):
    """Compute the jacobian of the configuration q.

    Parameters
    ----------
    q : np.ndarray
        Array of configuration of the robot, size rmodel.nq.

    Returns
    -------
    jacobian : np.ndarray
        Jacobian of the robot at the end effector at a configuration q, size 3 x rmodel.nq. 
    """
    # Computing the jacobian of the joints
    pin.computeJointJacobians(rmodel, rdata, q)

    # Computing the jacobien in the LOCAL_WORLD_ALIGNED coordonates system at the pose of the end effector.
    J = pin.getFrameJacobian(
        rmodel, rdata, EndeffID, pin.LOCAL_WORLD_ALIGNED)[:3]
    return J


def gradient_cost(q : np.ndarray):
    """Compute the gradient of the cost function at a configuration q.

    Parameters
    ----------
    q : np.ndarray
        Array of configuration of the robot, size rmodel.nq.

    Returns
    -------
    gradient cost : np.ndarray
        Gradient cost of the robot at the end effector at a configuration q, size rmodel.nq.
    """

    return np.dot(jacobian(q).T, residual(q))

def hessian(q: np.ndarray):
    """Returns hessian matrix of the end effector at a q position

    Parameters
    ----------
    q : np.ndarray
        Array of the configuration of the robot

    Returns
    -------
    Hessian matrix : np.ndaraay
        Hessian matrix at a given q configuration of the robot
    """
    pin.framesForwardKinematics(rmodel, rdata, q)
    return pin.computeFrameJacobian(rmodel, rdata, q, EndeffID, pin.LOCAL)


def callback(q : np.ndarray):
    vis.display(q)
    time.sleep(1e-2)

# Finite difference


def numdiff(f, x, eps=1e-6):
    """Estimate df/dx at x with finite diff of step eps

    Parameters
    ----------
    f : function handle
        Function evaluated for the finite differente of its gradient.
    x : np.ndarray
        Array at which the finite difference is calculated
    eps : float, optional
        Finite difference step, by default 1e-6

    Returns
    -------
    jacobian : np.ndarray
        Finite difference of the function f at x.
    """

    xc = x.copy()
    f0 = copy.copy(f(x))
    res = []
    for i in range(len(x)):
        xc[i] += eps
        res.append(copy.copy(f(xc)-f0)/eps)
        xc[i] = x[i]
    return np.array(res).T


# Testing whether the jacobian is right
q = q0
Jd = numdiff(residual, q)
J = jacobian(q)
assert (np.linalg.norm(J-Jd) < 1e-5)

# Function visualizing the result 


# Solving the problem with a gradient descent

# gradient_descent = SolverWithDisplay(vis,cost, gradient_cost)
# results_GD = gradient_descent(q0)

input()
# Solving the problem with a newton's method 

newton_method = SolverWithDisplay(vis, cost, gradient_cost, hessian, step_type="newton")
results_NM = newton_method(q0)