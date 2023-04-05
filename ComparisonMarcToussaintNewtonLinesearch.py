import pinocchio as pin
import matplotlib.pyplot as plt 
import time
import numpy as np

from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from QuadraticProblemInverseKinematics import QuadratricProblemInverseKinematics
from Solver import Solver
from NewtonMethodMarcToussaint import NewtonMethodMt

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

    input()

    # Creating the QP
    QP = QuadratricProblemInverseKinematics(
        rmodel, rdata, gmodel, gdata, vis)

    # Initial configuration
    q0 = pin.randomConfiguration(rmodel)
    robot.q0 = q0

    # Setting the tolerance 

    eps = 1e-8

    # Displaying the initial configuration
    vis.display(q0)

     # Linesearch method 
    newton_method = Solver( QP.cost, QP.gradient_cost, QP.hessian,callback=callback,
                                    step_type="newton", bool_plot_results=False, eps=eps, verbose=True)
    
    results_NM = newton_method(q0)
    
    list_fval_nm, list_gradfkval_nm, list_alphak_nm = newton_method._fval_history, newton_method._gradfval_history, newton_method._alphak_history

    # Marc Toussaint method 

    # Going back to initial configuration
    q0 = pin.randomConfiguration(rmodel)
    robot.q0 = q0

    trust_region_solver = NewtonMethodMt(QP.cost, QP.gradient_cost, QP.hessian, callback=callback, verbose= True, bool_plot_results=False)
    res = trust_region_solver(q0)

    list_fval_mt, list_gradfkval_mt, list_alphak_mt, list_reguk = trust_region_solver._fval_history, trust_region_solver._gradfval_history, trust_region_solver._alphak_history, trust_region_solver._reguk_history

    plt.subplot(411)
    plt.plot(list_fval_mt, "-ob", label="Marc Toussaint's method")
    plt.plot(list_fval_nm, "-or", label="Newton method")
    plt.yscale("log")
    plt.ylabel("Cost")
    plt.legend()

    plt.subplot(412)
    plt.plot(list_gradfkval_mt, "-ob", label="Marc Toussaint's method")
    plt.plot(list_fval_nm, "-or", label="Newton method")
    plt.yscale("log")
    plt.ylabel("Gradient")
    plt.legend()

    plt.subplot(413)
    plt.plot(list_alphak_mt,  "-ob", label="Marc Toussaint's method")
    plt.plot(list_fval_nm, "-or", label="Newton method")
    plt.yscale("log")
    plt.ylabel("Alpha")
    plt.legend()

    plt.subplot(414)
    plt.plot(list_reguk, "-ob", label="Marc Toussaint's method")
    plt.yscale("log")
    plt.ylabel("Regularization")
    plt.xlabel("Iterations")
    plt.legend()

    plt.suptitle(
        " Comparison Newton's method and Marc Toussaint's Newton method")
    plt.show()
