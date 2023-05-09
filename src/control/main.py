import pinocchio as pin
import matplotlib.pyplot as plt
import time
import numpy as np
import hppfcl

from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper
from quadratic_problem_inverse_kinematics import QuadratricProblemInverseKinematics
from solver_trs import NewtonMethodMt
from utils import generate_reachable_target, numdiff


pin.seed(5)

def grad_numdiff(q: np.ndarray):
    return numdiff(QP.cost, q)

def hess_numdiff(q: np.ndarray):
    return numdiff(grad_numdiff, q)




def callback(q):
    vis.display(q)
    time.sleep(1e-3)

# Creation of the robot
robot_wrapper = RobotWrapper()
robot, rmodel, gmodel = robot_wrapper()
rdata = rmodel.createData()
gdata = gmodel.createData()

# Generating the target
TARGET = generate_reachable_target(rmodel, rdata)

# Generating the shape of the target 
# The target shape is a ball of 5e-2 radii at the TARGET position

TARGET_SHAPE = hppfcl.Sphere(5e-2)

# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(TARGET, robot=robot)

# Waiting for the user to press enter to start the optimization problem
input()

# Creating the QP
QP = QuadratricProblemInverseKinematics(
    rmodel, rdata, gmodel, gdata, TARGET, TARGET_SHAPE)

# Initial configuration
q0 = pin.randomConfiguration(rmodel)
robot.q0 = q0

# Displaying the initial configuration
vis.display(q0)

# Solving the problem
trust_region_solver = NewtonMethodMt(QP.cost, QP.grad, QP.hessian, callback=callback, verbose= True, bool_plot_results=True)
res = trust_region_solver(q0)

trust_region_solver = NewtonMethodMt(QP.cost, grad_numdiff, grad_numdiff, callback=callback, verbose= True, bool_plot_results=True)


res = trust_region_solver(q0)
