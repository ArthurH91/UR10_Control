import numpy as np

class LineSearch:
    def __init__(self, f, grad, hessf = None, alpha = 0.5, beta = 0.8, max_iter = 10000, eps = 1e-6, ls_type = "backtracking", step_type = "l2_steepest", cond = "Armijo"):
        """
        Initialize line search object with a function and its gradient.
        
        Args:
        - f: function to minimize
        - grad: gradient of the function
        - hessf (function, optional): Function handle of the hessian of the function to minimize. Defaults to None.
        - alpha: fraction to decrease step size
        - beta: factor to decrease function value
        - max_iter: maximum number of iterations
        - eps: tolerance for convergence
        - ls_type: type of linesearch : backtracking, zoom
        - step_type: Type of step ("newton", "l2_steepest", "l1_steepest", ...). Defaults to "l2_steepest"
        - cond: Armijo condition or Wolfe conditions
        """
        self.f = f
        self.grad = grad
        self.hessf = hessf
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.eps = eps
        self.ls_type = ls_type
        self.step_type = step_type
        self.cond = cond


    def set_wolfe_conditions_constants(self):
        return None

    def __call__(self, x0: np.ndarray, ns_solver = np.linalg.solve):
        """
        Performs a line search optimization algorithm on the function f.
        
        Args:
        - x: current point
        - ns_solver (function, optional): Function handle of the solver to use for solving the Newton step. Defaults to np.linalg.solve.
        
        Returns:
        x: Solution of the descent search.
        fval: Function value at the solution of the search.
        gradfval: Function gradient value at the solution of the search.
        """
        # Print header
        self.line_search_print_header()

        # Initialize counter of iterations
        cnt_iter = 0
        # Initialize guess
        x = x0
        # Initialize stepsize
        step_size_k = self.alpha

        # Start
        while True:

            # Evaluate function
            fval_k = self.f(x)
            # Evaluate gradient
            gradfval_k = self.grad(x)
            # Evaluate hessian if step_type = newton
            if self.step_type == "newton":
                hessfval_k = self.hessf(x)
            else:
                hessfval_k = None 
            # Evaluate norm of gradient
            norm_gradfval_k = np.linalg.norm(gradfval_k)

            # Print
            self.line_search_print_iteration(
                cnt_iter, fval_k, norm_gradfval_k, step_size_k)
            # Every 30 iterations print header
            if cnt_iter % 30 == 29:
                self.line_search_print_header()

            # Check stopping conditions
            if self.line_search_convergence_criterion(norm_gradfval_k, self.eps) or self.line_search_exceeded_iterations(cnt_iter, self.max_iter):
                break

            # Compute search direction
            search_dir_k = self.line_search_compute_search_direction(
                fval_k, gradfval_k, hessfval_k, ns_solver)

            # Choose stepsize
            step_size_k = self.line_search_choose_stepsize(fval_k, gradfval_k, x, search_dir_k)

            # Update solution
            x = x + step_size_k * search_dir_k

            # Update iteration counter
            cnt_iter += 1

        # Print output message
        self.line_search_print_output(cnt_iter, fval_k, norm_gradfval_k)

        # Return
        return x, fval_k, gradfval_k

    
    def line_search_exceeded_iterations(self, cnt_iter: int, max_iter: int):
        """Determines if the algorithm exceeded maximum iteration count.
        Args:
            cnt_iter (int): Iteration counter.
            max_iter (int): Maximum number of iterations.
        Returns:
            bool: Did the iteration counter exceed the maximum number of iterations?
        """
        return cnt_iter > max_iter
    
    def line_search_convergence_criterion(self, norm_gradfval_k: float, eps: float):
        """Determines if the algorithm attained the desired tolerance in satisfying the convergence conditions.

        Args:
            norm_gradfval_k (float): Norm of the gradient.
            eps (float): Epsilon tolerance.

        Returns:
            bool: Is the norm of the gradient smaller than the tolerance?
        """
        return norm_gradfval_k < eps

    
    def line_search_print_header(self):
        """Prints the header of the optimization textual output.
        """
        print("{:<7}".format("Iter.") + " | " + "{:^10}".format("f(x)") +
            " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))
    

    def line_search_print_iteration(self, cnt_iter: int, fval_k: float, norm_gradfval_k: float, step_size_k: float):
        """Prints the information of a single iteration of the gradient descent algorithm.
        Args:
            cnt_iter (int): Iteration counter.
            fval_k (float): Function value at current iterate.
            norm_gradfval_k (float): Gradient norm value at current iterate.
            step_size_k (float): Size of the step at current iterate.
        """
        print(("%7d" % cnt_iter) + " | " + ("%.4e" % fval_k) + " | " +
            ("%.4e" % norm_gradfval_k) + " | " + ("%.4e" % step_size_k))


    def line_search_print_output(self, cnt_iter: int, fval_k: float, norm_gradfval_k: float):
        """Prints the final message of the search.
        Args:
            cnt_iter (int): Iteration counter.
            fval_k (float): Function value at current iterate.
            norm_gradfval_k (float): Gradient norm value at current iterate.
            eps (float): Tolerance on the first order optimality condition for convergence.
            max_iter (int): Maximum number of iterations to perform.
        """
        # If the algorithm converged
        if self.line_search_convergence_criterion(norm_gradfval_k=norm_gradfval_k, eps= self.eps):
            print()
            print(f"The {self.step_type} method successfully converged in %d iterations." % cnt_iter)
            print("Optimal function value: %.4e." % fval_k)
            print("Optimality conditions : %.4e." % norm_gradfval_k)

        # If the algorithm exceeded the iterations
        if self.line_search_exceeded_iterations(cnt_iter=cnt_iter, max_iter=self.max_iter):
            print()
            print("Gradient descent exceeded the maximum number (%d) of iterations." % self.max_iter)
            print("Current function value: %.4e." % fval_k)
            print("Current optimality conditions : %.4e." % norm_gradfval_k)


    def line_search_compute_search_direction(self, fval_k: float, gradfval_k: np.ndarray, hessfval_k = [], ns_solver=np.linalg.solve):
        """Computes the search direction for the line search.

        Args:
            fval_k (float): Function value at current iterate.
            gradfval_k (np.ndarray): Function gradient value at current iterate.
            hessfval_k (np.ndarray, optional): Value of the hessian fonction at the current iteration. Defaults to [].
            ns_solver (_type_, optional): Defaults to np.linalg.solve.
        """
        # If we want the steepest descent direction in the L2 norm
        if self.step_type == "l2_steepest":
            return -gradfval_k
        if self.step_type == "newton":
            return ns_solver(hessfval_k, - gradfval_k)
        if self.step_type == "bfgs":
            return 
    
    def line_search_armijo_condition_is_true(self, fval_k, gradfval_k, x, alpha, search_dir_k):
        """Check whether the Armijo condition is respected.


        Parameters
        ----------
        fval_k : np.ndarray
            Value of f 
        gradfval_k : _type_
            _description_
        x : _type_
            _description_
        search_dir_k : _type_
            _description_

        Returns
        -------
        Boolean
            True if Armijo condition is respected.
        """
        # Armijo number
        c = 0.8
        # Calculate the value of the function at the next iter
        fnext = self.f(x + alpha * search_dir_k)
        # Calculate the directional derivative along the search direction
        dirderiv = alpha * np.dot(gradfval_k, search_dir_k)

        # Return Armijo condition
        return fnext < fval_k + c * dirderiv

    def line_search_choose_stepsize(self, fval_k : float, gradfval_k : np.ndarray, x : np.ndarray, search_dir_k : np.ndarray):
        """Chooses the stepsize of the line search.

        Args:
            f (_type_): _description_
            gf (_type_): _description_
            fval_k (float): _description_
            gradfval_k (np.ndarray): _description_
            x (np.ndarray): _description_
            search_dir_k (np.ndarray): _description_
            alpha (float): _description_
            ls_type (str, optional): _description_. Defaults to "backtracking".
        """
        # When the type is backtracking
        if self.ls_type == "backtracking":

            # Choose ro
            ro = 0.7
            # Initialize the step iterate
            alpha = self.alpha
            
            # Repeat
            while True:

                # Check stopping conditions
                if self.line_search_armijo_condition_is_true(fval_k, gradfval_k, x, alpha, search_dir_k):
                    break

                # Otherwise diminish alpha
                alpha = ro * alpha
            
            # Return
            return alpha
            
    

if __name__ == "__main__":

    X = []