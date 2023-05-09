import pinocchio as pin
import matplotlib.pyplot as plt 
import time

from create_visualizer import create_visualizer
from robot_wrapper import RobotWrapper
from Solver import Solver
from quadratic_problem_inverse_kinematics import QuadratricProblemInverseKinematics

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

    eps = 1e-8   

    # Solving the problem with a gradient descent
    gradient_descent = Solver(QP.cost, QP.gradient_cost, callback=callback, bool_plot_results=True, eps=eps, max_iter = 1000)
    results_GD = gradient_descent(q0)
    list_q_gd = gradient_descent._gradfval_history

    # Solving the problem with a newton's method
    newton_method = Solver( QP.cost, QP.gradient_cost, QP.hessian,callback=callback,
                                    step_type="newton", bool_plot_results=True, eps=eps, verbose=True)
    results_NM = newton_method(q0)
    list_q_nm = newton_method._gradfval_history

    # Comparing the results

    plt.plot(list_q_gd,"-o", label = "Gradient descent")
    plt.plot(list_q_nm, "-o", label = "Gauss Newton method")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Value of the gradient")
    plt.title('Minimization of the gradient')
    plt.yscale("log")
    plt.show()
