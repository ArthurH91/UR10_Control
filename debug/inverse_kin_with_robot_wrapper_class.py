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


def cost(q: np.ndarray):
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


def jacobian(q: np.ndarray):
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


def gradient_cost(q: np.ndarray):
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


def callback(q: np.ndarray):
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

# Functions solving the problem :


def BFGS(q0: np.ndarray, ALPHA=.1, MAX_ITER=50, plot=False):
    """Optimize the configuration of the robot to reach the target.

    Parameters
    ----------
    q0 : np.ndarray
        Initial configuration of the robot.
    ALPHA : float, optional
        Step of the descent, by default .1
    MAX_ITER : int, optional
        Number max of iterations, by default 50
    plot : bool, optional
        plotting the cost function through iterations, by default False

    Returns
    -------
    res_list : list
        List of the residuals through iterations
    q_list : list
        List of the configurations through iterations
    cost_list : list
        List of the cost through iterations
    """

    # Creating lists for storing the results and configurations
    res_list = []
    q_list = [q0]
    cost_list = []

    q = q0

    for i in range(MAX_ITER):

        # Computing the residuals at the configuration q.
        res = residual(q)

        # Computing the jacobian at the end effector pose.
        J = jacobian(q)

        # Computing the cost value of the configuration q.
        costval = cost(q)

        # Computing the next step of q.
        q -= ALPHA * np.linalg.pinv(J)@res  # BFGS

        # Filling the lists
        res_list.append(np.linalg.norm(res))
        q_list.append(q)
        cost_list.append(costval)

        # Printing the outputs
        print(f" || {i} | {costval} ||")

        # Updating the visualizer
        callback(q)

    if plot:
        # Plotting the results
        plt.plot(cost_list)
        plt.xlabel("iterations")
        plt.ylabel("cost value")
        plt.title("Cost through the iterations")
        plt.show()

    return res_list, q_list, cost_list

# Solving the problem


# Step coefficient
ALPHA = .1

MAX_ITER = 50

res_list, q_list, cost_list = BFGS(q0, ALPHA, MAX_ITER, True)
