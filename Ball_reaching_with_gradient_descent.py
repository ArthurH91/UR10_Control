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
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
from create_visualizer import create_visualizer
from RobotWrapper import RobotWrapper
from OptimizationProblem import OptimizationProblem
from Solver import Solver

# The goal of this program is to optimize the position of an UR10 robot reaching a ball from a random initial position.


# Computing the cost function, the gradient and the hessian of the problem.

def cost_function(q): return optimization_problem.compute_cost_function(q)
def gradient_cost_function(q): return optimization_problem.compute_gradient_cost_function(q)
def hessian(q): return optimization_problem._compute_hessian(q)

# Subclass of the solver including pinocchio and the model.

class SolverWithDisplay(Solver):
    def __init__(self,vis, f, grad, hess=None, alpha=0.5, alpha_max=10, beta=0.8, max_iter=1000, eps=0.000001, ls_type="backtracking", step_type="l2_steepest", cond="Armijo", armijo_const=0.0001, wolfe_curvature_const=0.8, lin_solver=np.linalg.solve, bool_plot_cost_function = False):
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
        super().__init__(f, grad, hess, alpha, alpha_max, beta, max_iter, eps, ls_type, step_type, cond, armijo_const, wolfe_curvature_const, lin_solver, bool_plot_cost_function)
        self.vis = vis


    def _compute_next_step(self):
            """Computes the next step of the iterations.

            Returns
            -------
            xval_k+1
                The next value of xval_k, depending on the search direction and the alpha found by the linesearch.
            """
            next_xval = self._xval_k + self._alpha_k * self._search_dir_k
            optimization_problem.callback(next_xval)
            return next_xval

if __name__ == "__main__":

    # Getting rid of the randomness of the problem
    pin.seed(0)

    # Unwrapping the robot and adding the sphere as a target
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    
    # Creating the datas of the models
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Creating the visualizer of the robot
    vis = create_visualizer(robot)

    # Creating a random configuration that will be used for the initial position of the robot
    q = pin.randomConfiguration(rmodel)

    # Creating the object allowing the computation of the cost, gradient and hessian of the optimization problem
    optimization_problem = OptimizationProblem(rmodel, rdata, gmodel, gdata, vis)

    # Waiting for the user's input to start the script
    input()

    # Creating an object Solver that will be used to solve the optimization problem
    newton_method = SolverWithDisplay(vis,cost_function, gradient_cost_function, hessian, max_iter=50, alpha = 1, beta = 0.5, bool_plot_cost_function= True, step_type="newton")
    results = newton_method(q)

    print(f"The final q is : {newton_method._xval_history[-1]}")

    # Start of the debug : 

    # The idea here is to make the robot reach the ball with the configuration used to determine the position of the target.
    # In order to have only reachable positions for the ball, a random configuration is generated for the robot at its initialization 
    # and the target is placed at this pose.   

    input()
    # SE3 of the target
    M_target = robot_wrapper._M_target
    print(f' The SE3 of the target is : {M_target}')

    # Configuration vector to attain the target with the end effector
    q_target = robot_wrapper._q_target
    print(f'A configuration that can be used to reach the target : {q_target} \n')

    # Going to the target
    pin.framesForwardKinematics(rmodel, rdata, q_target)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q_target)

    # Visualizing the configuration
    print("Verification of the cost function and its gradient.")
    print("The following cost and gradient should be near 0 as the distance between the target and the end effector is approching 0.")
    vis.display(q_target)
    print(f'cost_function(q_target) : {cost_function(q_target)}')
    print(f'gradient_cost_function(q_target) : {gradient_cost_function(q_target)}')
    print(f'Distance between the frames end effector and target when going to q_target : {np.linalg.norm(optimization_problem._compute_vector_between_two_frames("endeff_geom", "target_geom", q_target))}')
