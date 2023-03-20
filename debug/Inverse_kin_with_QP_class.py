import pinocchio as pin
import matplotlib.pyplot as plt 
import time

from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from Solver import Solver
from QuadraticProblemInverseKinematics import QuadratricProblemInverseKinematics

def callback(q):
    vis.display(q)
    time.sleep(1e-3)

if __name__ == "__main__":

    # Creation of the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Open the viewer
    vis = create_visualizer(robot)

    # Creating the QP
    QP = QuadratricProblemInverseKinematics(rmodel, rdata, gmodel, gdata, vis)

    # Initial configuration
    # pin.seed(0)
    q0 = pin.randomConfiguration(rmodel)
    robot.q0 = q0

    vis.display(q0)

    # Function visualizing the result

    eps = 1e-4

    # Solving the problem with a gradient descent
    gradient_descent = Solver(callback, QP.cost, QP.gradient_cost, bool_plot_cost_function=True, eps=eps, max_iter = 1000)
    results_GD = gradient_descent(q0)
    list_q_gd = gradient_descent._fval_history

    # Solving the problem with a newton's method
    newton_method = Solver(callback, QP.cost, QP.gradient_cost, QP.hessian,
                                    step_type="newton", bool_plot_cost_function=True, eps=eps, verbose=True)
    results_NM = newton_method(q0)
    list_q_nm = newton_method._fval_history

    # Comparing the results

    # To compare the results, having the same number of elements in the list is mandatory 
    if len(list_q_nm) < len(list_q_gd):
        while len(list_q_nm) < len(list_q_gd):
            list_q_nm.append(0)
    else:
        while len(list_q_gd) < len(list_q_nm):
            list_q_gd.append(0)
    plt.plot(list_q_gd,"-o", label = "Gradient descent")
    plt.plot(list_q_nm, "-o", label = "Gauss Newton method")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Value of the cost function")
    plt.title('Minimization of the cost function')
    plt.show()
