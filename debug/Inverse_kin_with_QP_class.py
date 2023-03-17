import pinocchio as pin

from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from Solver import SolverWithDisplay
from QuadraticProblemInverseKinematics import QuadratricProblemInverseKinematics


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
    pin.seed(0)
    q0 = pin.randomConfiguration(rmodel)
    robot.q0 = q0

    vis.display(q0)

    # Function visualizing the result

    eps = 1e-4

    # Solving the problem with a gradient descent
    gradient_descent = SolverWithDisplay(
        vis, QP.cost, QP.gradient_cost, bool_plot_cost_function=True, time_sleep=1e-3, eps=eps)
    results_GD = gradient_descent(q0)

    # Solving the problem with a newton's method
    newton_method = SolverWithDisplay(vis, QP.cost, QP.gradient_cost, QP.hessian,
                                    step_type="newton", bool_plot_cost_function=True, time_sleep=0.1, eps=eps)
    results_NM = newton_method(q0)
