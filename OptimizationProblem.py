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
from RobotWrapper import RobotWrapper
from create_visualizer import create_visualizer

# This class is for defining the optimization problem and computing the cost function, its gradient and hessian.

class OptimizationProblem():
    def __init__(self, rmodel: pin.Model,rdata: pin.Data, gmodel: pin.GeometryModel, gdata: pin.GeometryData, vis):
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


    def get_transform_from_Frame(self, frame_geom: str):
        """returns the transform given a frame name, takes the str name of the frame.

        Parameters
        ----------
        frame_geom : str
            Name of the geometrical frame

        Returns
        -------
        SE3 Vector
            Transform of the given frame in SE3
        """
        frame_Id = self._gmodel.getGeometryId(frame_geom)
        return self._gdata.oMg[frame_Id]
    

    def _compute_vector_between_two_frames(self, frame_geom1: str, frame_geom2: str, q: np.ndarray):
        """ Computes the the vector linking the 2 geometrical frames. They HAVE to be geometrical frames.

        Parameters
        ----------
        frame_geom1 : str
            1st geometrical frame
        frame_geom2 : str
            2nd geometrical frame

        Returns
        -------
        dist : float
            Distance between the two frames
        """
        # Updating the geomtry placements is mandatory here lest the model isn't updated
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)
        pin.updateGeometryPlacements(self._rmodel, self._rdata, self._gmodel, self._gdata, q)

        # Getting the 2 poses of the frames 
        pose_frame1 = self.get_transform_from_Frame(frame_geom1)
        pose_frame2 = self.get_transform_from_Frame(frame_geom2)

        # Multiplying one frame by the inverse of the other to obtain the distance
        dist = pose_frame1 * pose_frame2.inverse()
        return dist.translation
    
    def compute_cost_function(self, q: np.ndarray):
        """Computes the quadratic cost function, that is the squared distance between the target and the end-effector.
        The form of the quadratic function is : f(q) = 1/2 * x^T @ x.
        Where here, x is the vector linking the frame of the target and the one of the end effector.
        Parameters 
        ----------
        q : np.ndarray
            Array of the configuration of the robot

        Returns
        -------
        cost : float
            Square distance between the target and the end effector at the configuration q 
        """
        dist = self._compute_vector_between_two_frames("endeff_geom", "target_geom",q)
        return 0.5 * np.linalg.norm(dist) ** 2 
    

    def compute_gradient_cost_function(self, q: np.ndarray):
        """Computes the gradient of the cost function. As the cost function is quadratic and has the following form : 
        f(q) = 1/2 * x^T @ x, the gradient is 

        Parameters
        ----------
        q : np.ndarray
            Array of the configuration of the robot

        Returns
        -------
        _type_
            Returns the gradient of the cost function.
        """

        # Computing the distance vector between the end effector geometrical frame and the target geometrical frame.
        dist = self._compute_vector_between_two_frames(
            "endeff_geom", "target_geom", q)

        # Computing the Jacobian at the frame of the end effector. 
        jacobian = pin.computeFrameJacobian(
            self._rmodel, self._rdata, q, self._rmodel.getFrameId("endeff"), pin.WORLD)[:3, :]
        return np.dot(jacobian.transpose(), dist)
    
    def _compute_hessian(self, q: np.ndarray):
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
        return pin.computeFrameJacobian(self._rmodel, self._rdata, q, self._rmodel.getFrameId("endeff"), pin.WORLD)

    def callback(self, q: np.ndarray):
        self._vis.display(q)

    def _update_robot(self, q: np.ndarray):
        """Updates the models and the datas of the robot.

        Parameters
        ----------
        q : np.ndarray
            Array of the configuration of the robot
        """
        pin.framesForwardKinematics(self._rmodel, self._rdata, q)
        pin.updateGeometryPlacements(self._rmodel, self._rdata, self._gmodel, self._gdata, q)

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

    rob = OptimizationProblem(rmodel, rdata, gmodel, gdata, vis)
    grad = rob.compute_gradient_cost_function(q) 
    print(grad)