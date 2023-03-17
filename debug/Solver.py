# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import time

class Solver:
    def __init__(self, f, grad, hess=None, alpha=0.5, alpha_max=10, beta=0.8, max_iter=1e3, eps=1e-6, ls_type="backtracking", step_type="l2_steepest", cond="Armijo", armijo_const=1e-4, wolfe_curvature_const=0.8, lin_solver=np.linalg.solve, bool_plot_cost_function=False):
        """Initialize solver object with the cost function and its gradient, along with numerical and categorical parameters.

        Args:
            f (function handle): Function handle of the minimization objective function.
            grad (function handle): Function handle of the gradient of the minimization objective function.
            hess (function handle, optional): Function handle of the hessian of the minimization objective function. Defaults to None.
            alpha (float, optional): Initial guess for step size. Defaults to 0.5.
            alpha_max (float, optional): Maximum step size. Defaults to 10.
            beta (float, optional): Default decrease on backtracking. Defaults to 0.8.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1e3.
            eps (float, optional): Tolerance for convergence. Defaults to 1e-6.
            ls_type (str, optional): Linesearch type. Defaults to "backtracking".
            step_type (str, optional): Type of step ("newton", "l2_steepest", "l1_steepest", ...). Defaults to "l2_steepest".
            cond (str, optional): Conditions to check at each iteration. Defaults to "Armijo".
            armijo_const (float, optional): Constant in the checking of Armijo condition. Defaults to 1e-4.
            wolfe_curvature_const (float, optional): Constant in the checking of the stong Wolfe curvature condition. Defaults to 0.8.
            lin_solver (function handle, optional): Solver for linear systems. Defaults to np.linalg.solve.
            bool_plot_cost_function (bool, optional): Boolean determining whether the user wants to print a plot of the cost function, by default False
        """

        # Initialize the object
        self._f = f
        self._grad = grad
        self._hess = hess
        self._alpha = alpha
        self._alpha_max = alpha_max
        self._beta = beta
        self._max_iter = max_iter
        self._eps = eps
        self._ls_type = ls_type
        self._step_type = step_type
        self._cond = cond
        self._armijo_const = armijo_const
        self._wolfe_curvature_const = wolfe_curvature_const
        self._lin_solver = lin_solver
        self._bool_plot_cost_function = bool_plot_cost_function


    def set_wolfe_conditions_constants(self):
        return None

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
            self._xval_k = self._compute_next_step()
            # Update iteration counter
            self._iter_cnter += 1

            # Adding the cost function to the history
            self._f_val_history.append(self._fval_k)

            # Adding the current solution to the history
            self._xval_history.append(self._xval_k)

        # Print output message
        self._print_output()

        # If the user wants to plot the cost function
        if self._bool_plot_cost_function:
            self._plot_cost_function()

        # Return
        return self._xval_k, self._fval_k, self._gradval_k

    def _exceeded_maximum_iterations(self):
        """Determines if the algorithm exceeded maximum iteration count.

        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._max_iter (int): Maximum number of iterations.

        Returns:
            bool: Did the iteration counter exceed the maximum number of iterations?
        """
        return self._iter_cnter > self._max_iter

    def _convergence_condition(self):
        """Determines if the algorithm attained the desired tolerance in satisfying the convergence conditions.

        Properties used:
            self._norm_gradfval_k (float): Norm of the gradient.
            self._eps (float): Convergence tolerance.

        Returns:
            bool: Is the norm of the gradient smaller than the tolerance?
        """
        return self._norm_gradfval_k < self._eps

    def _print_header(self):
        """Prints the header of the optimization textual output.
        """
        print("{:<7}".format("Iter.") + " | " + "{:^10}".format("f(x)") +
              " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))

    def _print_iteration(self):
        """Prints the information of a single iteration of the gradient descent algorithm.

        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._fval_k (float): Function value at current iterate.
            self._norm_gradfval_k (float): Gradient norm value at current iterate.
            self._alpha_k (float): Size of the step at current iterate.
        """
        print(("%7d" % self._iter_cnter) + " | " + ("%.4e" % self._fval_k) + " | " +
              ("%.4e" % self._norm_gradfval_k) + " | " + ("%.4e" % self._alpha_k))

    def _print_output(self):
        """Prints the final message of the search.

        Properties used:
            self._iter_cnter (int): Iteration counter.
            self._fval_k (float): Function value at current iterate.
            self._norm_gradfval_k (float): Gradient norm value at current iterate.
        """
        # If the algorithm converged
        if self._convergence_condition():
            print()
            print(self._step_type, "descent successfully converged in %d iterations." %
                  self._iter_cnter)
            print("Optimal function value: %.4e." % self._fval_k)
            print("Optimality conditions : %.4e." % self._norm_gradfval_k)

        # If the algorithm exceeded the iterations
        if self._exceeded_maximum_iterations():
            print()
            print(self._step_type,
                "descent exceeded the maximum number (%d) of iterations." % self._max_iter)
            print("Current function value: %.4e." % self._fval_k)
            print("Current optimality conditions : %.4e." %
                  self._norm_gradfval_k)

    def _compute_current_directional_derivative(self):
        """Computes the directional derivative at current iterate.
        Returns:
            float: Directional derivative at current iterate.
        """
        # Product between current gradient and current search direction
        return np.dot(self._gradval_k, self._search_dir_k)

    def _compute_directional_derivative(self, xval: np.ndarray, search_dir: np.ndarray):
        """Computes the directional derivative at a given point, for a given direction.
        Args:
            xval (np.ndarray): Point at which to calculate the directional derivative.
            search_dir (np.ndarray): Direction along which to calculate the directional derivative.
        Returns:
            float: Directional derivative at current iterate.
        """
        # Compute gradient at the point
        gradval = self._grad(xval)
        # Return directional derivative as the dot product between the gradient
        # and the search direction
        return np.dot(gradval, search_dir)

    def _compute_search_direction(self):
        """Computes the search direction for the line search.

        Properties used:
            self._step_type (string): Type of search direction to compute.
            self._gradval_k (np.ndarray): Function gradient value at current iterate.
            self._hessval_k (np.ndarray, optional): Value of the hessian fonction at the current iteration. Defaults to None.

            self._lin_solver (function handle, optional): Solver for linear systems. Defaults to np.linalg.solve.
        """
        # If we want the steepest descent direction in the L2 norm
        if self._step_type == "l2_steepest":
            return -self._gradval_k
        if self._step_type == "newton":
            try:
                return self._lin_solver(self._hessval_k, -self._gradval_k)
            except:
                print("Warning, the hessian matrix cannot be inversed. Pseudo inverse used here")
                return - np.linalg.pinv(self._hessval_k) @ self._gradval_k
        if self._step_type == "bfgs":
            return self._compute_bfgs_step()

    def _armijo_condition_is_true(self, alpha, fval_alpha):
        """Check whether the Armijo condition is respected for a given step size, and function value corresponding to the step size. 
        Args:
            alpha (float): Value of step-size for which to check the condition.
            fval_alpha (float): Value of function at the point given by the stepsize.
        Properties used:
            self._f (function handle): Function handle of the minimization objective function.
            self._xval_k (np.ndarray): Value of the current iterate.
            self._search_dir_k (np.ndarray): The search direction at current iterate.
            self._dir_deriv_k (float): The directional derivative of the function along current search direction.        
        Returns
            bool: True if Armijo condition is respected.
        """
        # Return truth value of Armijo condition
        return fval_alpha <= self._fval_k + self._armijo_const * alpha * self._dir_deriv_k

    def _compute_stepsize(self):
        """Chooses the stepsize of the line search.

        Args:
            ls_type (str, optional): _description_. Defaults to "backtracking".
        """
        # When the type is backtracking
        if self._ls_type == "backtracking":
            # Call the backtracking method
            return self._backtracking()
        # When the type is advanced
        if self._ls_type == "advanced":
            # Call the advanced line search algorithm
            return self._advanced_line_search()

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
        # Return
        return alpha
    
    def _compute_next_step(self):
        """Computes the next step of the iterations.

        Returns
        -------
        xval_k+1
            The next value of xval_k, depending on the search direction and the alpha found by the linesearch.
        """

        return self._xval_k + self._alpha_k * self._search_dir_k

    def _plot_cost_function(self):
        try: 
            import matplotlib.pyplot as plt
            plt.plot(self._f_val_history, "-o")
            plt.xlabel("Iterations")
            plt.ylabel("Value of the cost function")
            plt.title("Plot of the value of the cost function through the iterations")
            plt.show()
        except:
            print("No module named matplotlib.pyplot") 


class SolverWithDisplay(Solver):
    def __init__(self, vis, f, grad, hess=None, alpha=0.5, alpha_max=10, beta=0.8, max_iter=1000, eps=0.000001, ls_type="backtracking", step_type="l2_steepest", cond="Armijo", armijo_const=0.0001, wolfe_curvature_const=0.8, lin_solver=np.linalg.solve, bool_plot_cost_function=False, time_sleep = 0):
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
        bool_plot_cost_function : bool, optional
            Boolean determining whether the user wants to print a plot of the cost function, by default False
        """
        super().__init__(f, grad, hess, alpha, alpha_max, beta, max_iter, eps, ls_type, step_type,
                         cond, armijo_const, wolfe_curvature_const, lin_solver, bool_plot_cost_function)
        self._vis = vis
        self._time_sleep = time_sleep

    def _compute_next_step(self):
        """Computes the next step of the iterations.

            Returns
            -------
            xval_k+1
                The next value of xval_k, depending on the search direction and the alpha found by the linesearch.
            """
        next_xval = self._xval_k + self._alpha_k * self._search_dir_k
        self._vis.display(next_xval)
        time.sleep(self._time_sleep)
        return next_xval

if __name__ == "__main__":

    X = []
