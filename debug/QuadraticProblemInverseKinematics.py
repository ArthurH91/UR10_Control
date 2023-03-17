import numpy as np
import pinocchio as pin
import copy

from RobotWrapper import RobotWrapper
from create_visualizer import create_visualizer

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class QuadratricProblemInverseKinematics():
    def __init__(self, rmodel: pin.Model, rdata: pin.Data, gmodel: pin.GeometryModel, gdata: pin.GeometryData, vis):
        """Initialize the class with the models and datas of the robot.

        Parameters
        ----------
        _rmodel : pin.Model
            Model of the robot
        _rdata : pin.Data
            Data of the model of the robot
        _gmodel : pin.GeometryModel
            Geometrical model of the robot
        _gdata : pin.GeometryData
            Geometrical data of the model of the robot
        q : np.ndarray
            Array of configuration of the robot, size robot.nq
        _vis : meschat.Visualizer
            Visualizer used for displaying the robot.
        """
        self._rmodel = rmodel
        self._rdata = rdata
        self._gmodel = gmodel
        self._gdata = gdata
        self._vis = vis

        # Storing the IDs of the frames of the end effector and the target

        self._TargetID = self._rmodel.getFrameId('target')
        assert (self._TargetID < len(self._rmodel.frames))

        self._EndeffID = self._rmodel.getFrameId('endeff')
        assert (self._EndeffID < len(self._rmodel.frames))

        # Storing the cartesian pose of the target
        self._target = self._rdata.oMf[self._TargetID].translation

    def residual(self, q: np.ndarray):
        """Compute residuals from a configuration q. 
        Here, the residuals are calculated by the difference between the cartesian position of the end effector and the target.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        residual : np.ndarray
            Array of the residuals at a configuration q, size 3. 
        """

        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)

        # Obtaining the cartesian position of the end effector.
        p = self._rdata.oMf[self._EndeffID].translation
        return (p - self._target)


    def cost(self, q: np.ndarray):
        """Compute the cost of the configuration q. The cost is quadratic here.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        cost : float
            Cost of the configuration q. 
        """
        return 0.5 * np.linalg.norm(self.residual(q))
    
    def jacobian(self, q: np.ndarray):
        """Compute the jacobian of the configuration q.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        jacobian : np.ndarray
            Jacobian of the robot at the end effector at a configuration q, size 3 x rmodel.nq. 
        """
        # Computing the jacobian of the joints
        pin.computeJointJacobians(self._rmodel, self._rdata, q)

        # Computing the jacobien in the LOCAL_WORLD_ALIGNED coordonates system at the pose of the end effector.
        J = pin.getFrameJacobian(
            self._rmodel, self._rdata, self._EndeffID, pin.LOCAL_WORLD_ALIGNED)[:3]
        
        return J
    
    def gradient_cost(self, q: np.ndarray):
        """Compute the gradient of the cost function at a configuration q.

        Parameters
        ----------
        q : np.ndarray
            Array of configuration of the robot, size rmodel.nq.

        Returns
        -------
        gradient cost : np.ndarray
            Gradient cost of the robot at the end effector at a configuration q, size rmodel.nq.
        """

        return np.dot(self.jacobian(q).T, self.residual(q))


    def hessian(self, q: np.ndarray):
        """Returns hessian matrix of the end effector at a q position

        Parameters
        ----------
        q : np.ndarray
            Array of the configuration of the robot

        Returns
        -------
        Hessian matrix : np.ndaraay
            Hessian matrix at a given q configuration of the robot
        """
        jacobian_val = self.jacobian(q)
        return jacobian_val.T @ jacobian_val
    
    
    def _numdiff(f, x, eps=1e-6):
        """Estimate df/dx at x with finite diff of step eps

        Parameters
        ----------
        f : function handle
            Function evaluated for the finite differente of its gradient.
        x : np.ndarray
            Array at which the finite difference is calculated
        eps : float, optional
            Finite difference step, by default 1e-6

        Returns
        -------
        jacobian : np.ndarray
            Finite difference of the function f at x.
        """

        xc = x.copy()
        f0 = copy.copy(f(x))
        res = []
        for i in range(len(x)):
            xc[i] += eps
            res.append(copy.copy(f(xc)-f0)/eps)
            xc[i] = x[i]
        return np.array(res).T


if __name__ == "__main__":

    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)

    q = pin.randomConfiguration(rmodel)
    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)
    vis.display(q)

    QP = QuadratricProblemInverseKinematics(rmodel, rdata, gmodel, gdata, vis)
    grad = QP.gradient_cost(q)
    print(grad)
