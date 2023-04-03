import pinocchio as pin
import matplotlib.pyplot as plt 
import time
import numpy as np

from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from QuadraticProblemInverseKinematics import QuadratricProblemInverseKinematics
from Solver import Solver

def callback(q):
    vis.display(q)
    time.sleep(1e-3)

class NewtonMethodMt(Solver):

    def __init__(self, f, grad, hess, max_iter=1e3, callback=None, alpha_increase=1.2, alpha_decrease =0.5, regularization_increase =1.0, regularization_decrease=0.5, armijo_const =1e-2, beta = 1e-2, init_regu =1e-9, alpha=1.0, verbose=True, lin_solver=np.linalg.solve, eps=1e-8):
        """_summary_

    Parameters
    ----------

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
    alpha_increase : float, optional
        increase of the alpha, by default 1.2
    alpha_decrease : float, optional
        decrease of the alpha, by default 0.5
    regularization_increase : float, optional
        increase of the damping, by default 1.0
    regularization_decrease : float, optional
        decrease of the damping, by default 0.5
    beta : _type_, optional
        constant c in the backtracking linesearch in Nocedal, by default 1e-2
    init_regu : float, optional
        intial damping, by default 1.0
    alpha : float, optional
        initial alpha, by default 1.0
    verbose : bool, optional
        boolean describing whether the user wants the verbose mode, by default True
    lin_solver : function, optional
        solver of the equation ax = b, by default np.linalg.solve
    eps : float, optional
        epserance used in the stopping criteria, by default 1e-8
    """
        

        self._f = f
        self._grad = grad
        self._hess = hess
        self._max_iter = max_iter
        self._callback = callback
        self._alpha_increase = alpha_increase
        self._alpha_decrease = alpha_decrease
        self._regularization_decrease = regularization_decrease
        self._regularization_increase = regularization_increase
        self._armijo_const = armijo_const
        self._beta = beta 
        self._init_regu = init_regu
        self._alpha = alpha
        self._verbose = verbose
        self._lin_solver = lin_solver
        self._eps = eps


    def __call__(self, x0 : np.ndarray):
        """_summary_

        Parameters
        ----------
        x0 : np.ndarray
            initial guess

        Returns
        -------
        WIP
            _description_
        """

        # Initialization of the step size 
        self._alpha_k = self._alpha 

        # Initialization of the damping
        self._regu_k = self._init_regu

        # Initial guess
        self._xval_k = x0

        # Initialize iteration counter
        self._iter_cnter = 0

        # Create a list for the values of cost function
        self._list_fval = []

        # Create a list for the values of the gradient function

        self._list_gradfval = []

        # Create a list for the values of step size

        self._list_alphak = []

        # Create a list for the values of the regularization 

        self._list_regularization_k = []

        # Printing a small explanation of the algorithm
        self._print_start()

        # Printing the header if the user wants a verbose solver
        if self._verbose:
            self._print_header()

        # Start
        while True:

            # Cost of the step
            self._fval_k = self._f(self._xval_k)
            # Gradient of the cost function 
            self._gradfval_k = self._grad(self._xval_k)
            # Norm of the gradient function 
            self._norm_gradfval_k = np.linalg.norm(self._gradfval_k)
            # Hessian of the cost function
            self._hessval_k = self._hess(self._xval_k)

            if self._verbose:
                # Print current iterate
                self._print_iteration()
                # Every 30 iterations print header
                if self._iter_cnter % 30 == 29:
                    self._print_header()

            # Check stopping conditions
            if self._convergence_condition() or self._exceeded_maximum_iterations():
                break


            # Linesearch
            self._Ls_bool = False
            while not self._Ls_bool:

                # Computing search direction
                self._search_dir_k = self._compute_search_direction()

                # Computing directionnal derivative   
                self._dir_deriv_k = self._compute_current_directional_derivative()


                # Linesearch, if the step is accepted Ls_bool = True and the alpha_k is kept. If it's not, the search direction is 
                # computed once again with an increase in the damping.
                self._alpha_k = self._backtracking()

                # If the step is not accepted, increase the damping
                if not self._Ls_bool:
                    self._regu_k *= self._regularization_increase

            # Computing next step
            self._xval_k = self._compute_next_step()

            self._alpha_k = min(self._alpha_increase* self._alpha_k, 1)

            # Updating the trust-region

            if self._alpha_k == 1:
                self._regu_k /= 10

            # Iterate the loop
            self._iter_cnter += 1

            # Adding the cost function value to the list
            self._list_fval.append(self._fval_k)

            # Adding the step size to the list 
            self._list_alphak.append(self._alpha_k)

            # Adding the value of the norm of the gradient to the list
            self._list_gradfval.append(self._norm_gradfval_k)

            # Adding the value of the regularization to the list
            self._list_regularization_k.append(self._regu_k)

        return self._xval_k, self._fval_k, self._gradfval_k


    def _backtracking(self):
        """Calculates a step using backtracking.

        Returns:
            float: Step value computed by the backtracking.
        """
        # Initialize the step iterate
        alpha = self._alpha
        # Repeat
        while True:
            # Caclulate current function value
            fval_curr = self._f(self._xval_k + alpha * self._search_dir_k)
            # Check stopping conditions
            if self._armijo_condition_is_true(alpha=alpha, fval_alpha=fval_curr):
                break
            # Otherwise diminish alpha
            alpha = self._beta * alpha

            # Trust region
            if alpha >= 2e-10:
                self._Ls_bool = False
                return alpha
        # Return
        self._Ls_bool = True
        return alpha
    

    def _compute_search_direction(self):
        
        return self._lin_solver(self._hessval_k + self._regu_k * np.eye(len(self._xval_k)), - self._gradfval_k)
    

    def _print_start(self):
        print("Start of the Newton method of Marc Toussaint")



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
    test = NewtonMethodMt(QP.cost, QP.gradient_cost, QP.hessian, callback=callback, verbose= True)

    res = test(q0)

    # Plotting the results 
    # plt.subplot(411)
    # plt.plot(list_fval)
    # plt.yscale("log")
    # plt.ylabel("Cost")
    # plt.xlabel("Iterations")
    # plt.title("Cost through the iterations")

    # plt.subplot(412)
    # plt.plot(list_gradfkval)
    # plt.yscale("log")
    # plt.ylabel("Gradient")
    # plt.xlabel("Iterations")
    # plt.title("Gradient through the iterations")

    # plt.subplot(413)
    # plt.plot(list_alphak)
    # plt.yscale("log")
    # plt.ylabel("Alpha")
    # plt.xlabel("Iterations")
    # plt.title("Alpha through the iterations")

    # plt.subplot(414)
    # plt.plot(list_regularization)
    # plt.yscale("log")
    # plt.ylabel("Regularization")
    # plt.xlabel("Iterations")
    # plt.title("Regularization through the iterations")

    # plt.suptitle("Newton method marc Toussaint")
    # plt.show()