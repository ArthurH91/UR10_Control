import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
from visualizer import create_visualizer
from robot_wrapper import robot_wrapper
from robot_optimization_problem import robot_optimization
from linesearch_class import LineSearch
# The goal of this program is to optimize the position of an UR10 robot reaching a ball from a random initial position.


def cost_function(q): return robot_optimization_object.compute_cost_function(q)
def gradient_cost_function(q): return robot_optimization_object.compute_gradient_cost_function(q)
def hessian(q): return robot_optimization_object._compute_hessian(q)

class Linesearch_with_display(LineSearch):

    def __init__(self,vis, f, grad, hess=None, alpha=0.5, alpha_max=10, beta=0.8, max_iter=1000, eps=0.000001, ls_type="backtracking", step_type="l2_steepest", cond="Armijo", armijo_const=0.0001, wolfe_curvature_const=0.8, lin_solver=np.linalg.solve, plot_cost_function = False):
        """ Supercharge the init method of linesearch to include a meshcat visualizer, for a pinocchio problem.

        Parameters
        ----------
        vis : Meshcat.Visualizer
            Visualizer used for displaying the robot.
        f : function handle
            Function handle of the minimization objective function.
        grad : function handle
            Function handle of the gradient of the minimization objective function.
        hess : _type_, optional
            Function handle of the hessian of the minimization objective function. Defaults to None, by default None
        alpha : float, optional
            Initial guess for step size. Defaults to 0.5, by default 0.5
        alpha_max : int, optional
            Maximum step size, by default 10
        beta : float, optional
            Default decrease on backtracking, by default 0.8
        max_iter : int, optional
            Maximum number of iterations, by default 1000
        eps : float, optional
            Tolerance for convergence, by default 0.000001
        ls_type : str, optional
            Linesearch type, by default "backtracking", full linesearch is to be implemented
        step_type : str, optional
            Type of step ("newton", "l2_steepest", "l1_steepest", ...), by default "l2_steepest"
        cond : str, optional
            Conditions to check at each iteration, by default "Armijo"
        armijo_const : float, optional
            Constant in the checking of Armijo condition, by default 0.0001
        wolfe_curvature_const : float, optional
            Constant in the checking of the stong Wolfe curvature condition, by default 0.8
        lin_solver : _type_, optional
            Solver for linear systems. Defaults to np.linalg.solve, by default np.linalg.solve
        plot_cost_function : bool, optional
            Boolean determining whether the user wants to print a plot of the cost function, by default False
        """
        super().__init__(f, grad, hess, alpha, alpha_max, beta, max_iter, eps, ls_type, step_type, cond, armijo_const, wolfe_curvature_const, lin_solver)
        self.vis = vis
        self.plot_cost_function = plot_cost_function

    def __call__(self, x0: np.ndarray):
        """
        Performs a line search optimization algorithm on the function f.
        
        Args:
        - x: current point
        
        Returns:
        x: Solution of the descent search.
        fval: Function value at the solution of the search.
        gradfval: Function gradient value at the solution of the search.
        """
        # Print header
        self._print_header()

        # Initialize guess
        self._xval_k = x0
        # Initialize iteration counter
        self._iter_cnter = 0
        # Initialize current stepsize
        self._alpha_k = self._alpha

        # Initialize a list used if plot_cost_function == True
        self._f_val_history = []

        # Initialize a list of the configurations of the robot
        self._xval_history = [x0]

        # Start
        while True:

            # Evaluate function
            self._fval_k = self._f(self._xval_k)
            # Evaluate gradient
            self._gradval_k = self._grad(self._xval_k)
            # Evaluate hessian if step_type = newton
            if self._step_type == "newton":
                self._hessval_k = self._hess(self._xval_k)
            # Update hessian if step_type = "bfgs"
            elif self._step_type == "bfgs":
                self._bfgs_update()
            # Else maintain a None pointer to the
            else:
                self._hessval_k = None

            # Evaluate norm of gradient
            self._norm_gradfval_k = np.linalg.norm(self._gradval_k)

            # Print current iterate
            self._print_iteration()
            # Every 30 iterations print header
            if self._iter_cnter % 30 == 29:
                self._print_header()

            # Check stopping conditions
            if self._convergence_condition() or self._exceeded_maximum_iterations():
                break

            # Compute search direction
            self._search_dir_k = self._compute_search_direction()

            # Compute directional derivative
            self._dir_deriv_k = self._compute_current_directional_derivative()

            # Choose stepsize
            self._alpha_k = self._compute_stepsize()

            # Update solution
            self._xval_k = self._xval_k + self._alpha_k * self._search_dir_k
            

            # Update iteration counter
            self._iter_cnter += 1

            # Displaying the result on the meshcat.Visualizer
            vis.display(self._xval_k)
            # Adding the cost function to the history
            self._f_val_history.append(self._fval_k)

            # Adding the current solution to the history
            self._xval_history.append(self._xval_k)
            # input()
        # Print output message
        self._print_output()

        # If the user wants to plot the cost function 
        if self.plot_cost_function:
            self._plot_cost_function()


        # Return
        return self._xval_k, self._fval_k, self._gradval_k
    
    def _plot_cost_function(self):
        plt.plot(self._f_val_history, "-o")
        plt.xlabel("Iterations")
        plt.ylabel("Value of the cost function")
        plt.title("Plot of the value of the cost function through the iterations")
        plt.show()


if __name__ == "__main__":

    # pin.seed(0)
    robot_wrapper_test = robot_wrapper()
    robot, rmodel, gmodel = robot_wrapper_test(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)
    q = pin.randomConfiguration(rmodel)

    robot_optimization_object = robot_optimization(rmodel, rdata, gmodel, gdata, vis)


    gradient_descent = Linesearch_with_display(vis,cost_function, gradient_cost_function, hessian, max_iter=10, alpha = 1, beta = 0.5, plot_cost_function= True, step_type="newton")
    test = gradient_descent(q)
    print(f"The q are the following : {gradient_descent._xval_history}")

    # print(gradient_descent._xval_history[0])

    # q0 = gradient_descent._xval_history[0]

    # test_dist = robot_optimization_object._compute_vector_between_two_frames("endeff_geom", "target_geom", q0)

    # print(f"Test dist : {test_dist}")

    input()
    M_target = robot_wrapper_test._M_target
    print(M_target)
    q_target = robot_wrapper_test._q_target
    print(q_target)
    pin.framesForwardKinematics(rmodel, rdata, q_target)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q_target)
    vis.display(q_target)
    print(cost_function(q_target))
    print(gradient_cost_function(q_target))
    print(np.linalg.norm(robot_optimization_object._compute_vector_between_two_frames("endeff_geom", "target_geom", q_target)))
    print(hessian(q).shape)