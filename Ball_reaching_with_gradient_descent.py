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
            vis.display(next_xval)
            return next_xval

if __name__ == "__main__":

    pin.seed(0)
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)
    q = pin.randomConfiguration(rmodel)

    optimization_problem = OptimizationProblem(rmodel, rdata, gmodel, gdata, vis)

    input()
    gradient_descent = SolverWithDisplay(vis,cost_function, gradient_cost_function, hessian, max_iter=20, alpha = 1, beta = 0.5, bool_plot_cost_function= True)
    test = gradient_descent(q)
    print(f"The q are the following : {gradient_descent._xval_history}")

    # print(gradient_descent._xval_history[0])

    # q0 = gradient_descent._xval_history[0]

    # test_dist = robot_optimization_object._compute_vector_between_two_frames("endeff_geom", "target_geom", q0)

    # print(f"Test dist : {test_dist}")

    input()
    M_target = robot_wrapper._M_target
    print(f'M_target : {M_target}')
    q_target = robot_wrapper._q_target
    print(f'q_target : {q_target}')
    pin.framesForwardKinematics(rmodel, rdata, q_target)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q_target)
    vis.display(q_target)
    print(f'cost_function(q_target) : {cost_function(q_target)}')
    print(f'gradient_cost_function(q_target) : {gradient_cost_function(q_target)}')
    print(f'Distance between the frames end effector and target when going to q_target (which is the ) : {np.linalg.norm(optimization_problem._compute_vector_between_two_frames("endeff_geom", "target_geom", q_target))}')
    print("--------------------------")
    # q = np.array([5.16500284, -4.67222658, 14.19583246, -2.21594461,  5.86370682,
    #               -3.78744479])
    # pin.framesForwardKinematics(rmodel, rdata, q_target)
    # pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)
    # vis.display(q)
    # print(f'cost_function(q) : {cost_function(q)}')
    # print(f'Distance between the frames end effector and target : {np.linalg.norm(robot_optimization_object._compute_vector_between_two_frames("endeff_geom", "target_geom", q))}')

