import numpy as np
import pinocchio as pin
import hppfcl

from wrapper_robot import RobotWrapper
import pydiffcol

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.


class QuadratricProblemInverseKinematics:
    def __init__(
        self,
        rmodel: pin.Model,
        rdata: pin.Data,
        gmodel: pin.GeometryModel,
        gdata: pin.GeometryData,
        target: np.ndarray,
        target_shape: hppfcl.ShapeBase,
    ):
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

        """
        self._rmodel = rmodel
        self._rdata = rdata
        self._gmodel = gmodel
        self._gdata = gdata
        self._vis = vis
        self._target = target
        self._target_shape = target_shape

        # Storing the IDs of the frames of the end effector and the target

        self._EndeffID = self._rmodel.getFrameId("endeff")
        self._EndeffID_geom = self._gmodel.getGeometryId("endeff_geom")
        assert self._EndeffID < len(self._rmodel.frames)
        assert self._EndeffID_geom < len(self._gmodel.geometryObjects)

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

        # Distance request for pydiffcol
        self._req = pydiffcol.DistanceRequest()
        self._res = pydiffcol.DistanceResult()

        # Forward kinematics of the robot at the configuration q.
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)

        # Obtaining the cartesian position of the end effector.
        self.endeff_Transform = self._rdata.oMf[self._EndeffID]
        self.endeff_Shape = self._gmodel.geometryObjects[self._EndeffID_geom].geometry

        residual = pydiffcol.distance(
            self.endeff_Shape,
            self.endeff_Transform,
            self._target_shape,
            self._target,
            self._req,
            self._res,
        )

        return 0.5 * np.linalg.norm(residual) ** 2

    # def jacobian(self, q: np.ndarray):
    #     """Compute the jacobian of the configuration q.

    #     Parameters
    #     ----------
    #     q : np.ndarray
    #         Array of configuration of the robot, size rmodel.nq.

    #     Returns
    #     -------
    #     jacobian : np.ndarray
    #         Jacobian of the robot at the end effector at a configuration q, size 3 x rmodel.nq.
    #     """
    #     # Computing the jacobian of the joints
    #     pin.computeJointJacobians(self._rmodel, self._rdata, q)

    #     # Computing the jacobien in the LOCAL_WORLD_ALIGNED coordonates system at the pose of the end effector.
    #     J = pin.getFrameJacobian(
    #         self._rmodel, self._rdata, self._EndeffID, pin.LOCAL_WORLD_ALIGNED
    #     )[:3]

    #     return J

    def grad(self, q: np.ndarray):
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
        self.cost(q)
        self._jacobian = pin.computeFrameJacobian(
            self._rmodel, self._rdata, q, self._EndeffID, pin.LOCAL
        )
        jacobian_transpose = self._jacobian.transpose()
        _ = pydiffcol.distance_derivatives(
            self.endeff_Shape,
            self.endeff_Transform,
            self._target_shape,
            self._target,
            self._req,
            self._res,
        )
        dw_dq_transpose = self._res.dw_dq.transpose()
        return jacobian_transpose @ dw_dq_transpose @ self._res.w

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

        self.grad(q)
        return self._jacobian.T @ self._jacobian


if __name__ == "__main__":
    from utils import generate_reachable_target
    from wrapper_meshcat import MeshcatWrapper

    # Creating the robot
    robot_wrapper = RobotWrapper()
    robot, rmodel, gmodel = robot_wrapper()
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Generating a target
    TARGET = generate_reachable_target(rmodel, rdata)

    # Generating an initial configuration
    q = pin.randomConfiguration(rmodel)
    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q)

    # Creating the visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(TARGET, robot=robot)

    # The target shape is a ball of 5e-2 radii at the TARGET position

    TARGET_SHAPE = hppfcl.Sphere(5e-2)

    QP = QuadratricProblemInverseKinematics(
        rmodel, rdata, gmodel, gdata, TARGET, TARGET_SHAPE
    )

    res = QP.cost(q)
    print(res)
    gradval = QP.grad(q)
    hessval = QP.hessian(q)
