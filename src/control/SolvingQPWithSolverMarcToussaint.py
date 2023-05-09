import pinocchio as pin
import matplotlib.pyplot as plt
import time
import numpy as np

from create_visualizer import create_visualizer
from robot_wrapper import RobotWrapper
from quadratic_problem_inverse_kinematics import QuadratricProblemInverseKinematics
from newton_method_marc_toussaint import NewtonMethodMt


def callback(q):
    vis.display(q)
    time.sleep(1e-3)

# Creation of the robot
robot_wrapper = RobotWrapper()
robot, rmodel, gmodel = robot_wrapper(target=True)
rdata = rmodel.createData()
gdata = gmodel.createData()

# Open the viewer
vis = create_visualizer(robot)

# Waiting for the user to press enter to start the optimization problem
input()

# Creating the QP
QP = QuadratricProblemInverseKinematics(
    rmodel, rdata, gmodel, gdata, vis)

# Initial configuration
q0 = pin.randomConfiguration(rmodel)
robot.q0 = q0

# Displaying the initial configuration
vis.display(q0)

# Solving the problem
trust_region_solver = NewtonMethodMt(QP.cost, QP.gradient_cost, QP.hessian, callback=callback, verbose= True, bool_plot_results=True)

res = trust_region_solver(q0)
