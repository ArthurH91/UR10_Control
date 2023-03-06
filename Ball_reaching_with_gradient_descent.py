import numpy as np
import pinocchio as pin
import example_robot_data as robex
import hppfcl
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
from visualizer import create_visualizer
from robot_wrapper import robot_wrapper
from robot_optimization_problem import robot_optimization
# The goal of this program is to optimize the position of an UR10 robot reaching a ball from a random initial position.


def cost_function(q): return robot_optimization_object.compute_cost_function(q)


if __name__ == "__main__":

    robot_wrapper_test = robot_wrapper()
    robot, rmodel, gmodel = robot_wrapper_test(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)
    q = pin.randomConfiguration(rmodel)

    robot_optimization_object = robot_optimization(rmodel, rdata, gmodel, gdata, vis)
    print(cost_function(q))