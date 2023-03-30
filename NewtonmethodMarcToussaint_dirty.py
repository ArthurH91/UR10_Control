import pinocchio as pin
import matplotlib.pyplot as plt 
import time
import numpy as np

from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from QuadraticProblemInverseKinematics import QuadratricProblemInverseKinematics


def callback(q):
    vis.display(q)
    time.sleep(1e-3)


def Newton_method_MT(x0: np.ndarray, f, grad, hess, max_iter=1e3, callback=None, varrho_alpha_plus=1.2, varrho_alpha_moins=0.5, varrho_lambda_plus=1.0, varrho_lambda_moins=0.5, varrho_ls=1e-2, initial_damping=1.0, alpha=1.0, verbose=True, lin_solver=np.linalg.solve, tol = 1e-8):
    """_summary_

    Parameters
    ----------
    x0 : np.ndarray
        initial guess
    f : function handle
        cost function
    grad : function handle
        gradient function of the cost function
    hess : function handle
        hessian function of the cost function
    max_iter : _type_, optional
        number max of iterations, by default 1e3
    callback : function handle, optional
        callback at each iteration, can be a display of meshcat for instanceription, by default None
    varrho_alpha_plus : float, optional
        increase of the alpha, by default 1.2
    varrho_alpha_moins : float, optional
        decrease of the alpha, by default 0.5
    varrho_lambda_plus : float, optional
        increase of the damping, by default 1.0
    varrho_lambda_moins : float, optional
        decrease of the damping, by default 0.5
    varrho_ls : _type_, optional
        constant c in the backtracking linesearch in Nocedal, by default 1e-2
    initial_damping : float, optional
        intial damping, by default 1.0
    alpha : float, optional
        initial alpha, by default 1.0
    verbose : bool, optional
        boolean describing whether the user wants the verbose mode, by default True
    lin_solver : function, optional
        solver of the equation ax = b, by default np.linalg.solve
    tol : float, optional
        tolerance used in the stopping criteria, by default 1e-8

    Returns
    -------
    WIP
        _description_
    """

    # Initialization of the step size 
    alpha_k = alpha 

    # Initialization of the damping
    damping = initial_damping

    # Initial guess
    xval_k = x0

    # Initialize iteration counter
    iter_cnter = 0

    # Create a list for the values of cost function
    list_fval = []

    # Start
    while True:
        # Cost of the step
        fval_k = f(xval_k)
        # Gradient of the cost function 
        gradval_k = grad(xval_k)
        # Norm of the gradient function 
        norm_gradval_k = np.linalg.norm(gradval_k)
        # Hessian of the cost function
        hessval_k = hess(xval_k)

        # Linesearch
        Ls_bool = False
        while not Ls_bool:

            # Computing search direction
            search_dir_k = lin_solver(hessval_k + damping * np.eye(len(xval_k)), - gradval_k)

            # Linesearch, if the step is accepted Ls_bool = True and the alpha_k is kept. If it's not, the search direction is 
            # computed once again with an increase in the damping.
            Ls_bool, alpha_k = backtracking(f, xval_k, alpha_k, search_dir_k, fval_k, gradval_k, varrho_ls, varrho_alpha_moins)

            # If the step is not accepted, increase the damping
            if not Ls_bool:
                damping *= varrho_lambda_plus

        # Computing next step
        xval_k += alpha_k * search_dir_k

        callback(xval_k)
        # Checking the convergence of the algorithm
        if convergence(norm_gradval_k, tol) or iter_cnter >= max_iter:
            break
        alpha_k = min(varrho_alpha_plus* alpha_k, 1)

        # Iterate the loop
        iter_cnter += 1

        # Adding the cost function value to the list
        list_fval.append(fval_k)

    return list_fval



def backtracking(f, xval_k: np.ndarray, alpha_k: float, search_dir_k: np.ndarray, fval_k: float ,gradval_k: np.ndarray, varrho_ls:float, varrho_alpha_moins: float):
    """Backtracking linesearchv, given a search direction, finds the approximative of the optimal step.

    Parameters
    ----------
    f : function handle
        cost function studied
    xval_k : np.ndarray
        Vector of input
    alpha_k : float
        stepsize 
    search_dir_k : np.ndarray
        direction 
    fval_k : float
        value of the cost function
    gradval_k : np.ndarray
        value of the gradient 
    varrho_ls : float
        coefficient c of Nocedal
    varrho_alpha_moins : float
        decrease of the alpha

    Returns
    -------
    Bool_ls : bool
        bool describing whether the backtracking converges
    alpha_k : float
        value of the stepsize
    """
    # Loose Wolfe condition
    while f(xval_k + alpha_k * search_dir_k) > fval_k + varrho_ls * alpha_k * np.dot(gradval_k, search_dir_k):
        # Decreasing stepsize
        alpha_k = varrho_alpha_moins * alpha_k

        # If the step length is too small, the step is refused.
        if alpha_k <= 2e-10:
            return False, alpha_k
    return True, alpha_k

def convergence(norm_gradval_k: float, tol: float):
    """Returns boolean whether the convergence criteria is met

    Parameters
    ----------
    norm_gradval_k : float
        norm of the gradient
    tol : float
        stopping criteria

    Returns
    -------
    convergence : bool
        boolean describing whether the convergence criteria is met
    """
    if norm_gradval_k < tol:
        return True
    return False



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

    # Displaying the initial configuration
    vis.display(q0)

    # 
    list_fval = Newton_method_MT(q0, QP.cost, QP.gradient_cost, QP.hessian, callback=callback)

    # Plotting the results 
    plt.plot(list_fval)
    plt.yscale("log")
    plt.ylabel("value of the cost function")
    plt.xlabel("Iterations")
    plt.title("Values of the cost function through the iterations")
    plt.show()