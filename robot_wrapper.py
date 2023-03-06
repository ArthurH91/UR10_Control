import numpy as np
import pinocchio as pin
import example_robot_data as robex
from visualizer import create_visualizer
import hppfcl

class robot_wrapper():

    def __init__(self, scale=1.0, name_robot="ur10"):
        """Initialize the wrapper with a scaling number of the target and the name of the robot wanted to get unwrapped.

        Parameters
        ----------
        scale : float, optional
            Scale of the target, by default 1.0
        name_robot : str, optional
            Name of the robot wanted to get unwrapped, by default "ur10"
        """

        self.scale = scale
        self.robot = robex.load(name_robot)
        self.rmodel = self.robot.model
        self.color = np.array([249, 136, 126, 255]) / 255

    def __call__(self, target=False):
        """Create a robot with a new frame at the end effector position and place a hppfcl: ShapeBase cylinder at this position.

        Parameters
        ----------
        target : bool, optional
            Boolean describing whether the user wants a target or not, by default False

        Returns
        -------
        robot
            Robot description of the said robot
        rmodel
            Model of the robot
        gmodel
            Geometrical model of the robot


        """

        # Creation of the frame for the end effector by using the frame tool0, which is at the end effector pose.
        # This frame will be used for the position of the cylinder at the end of the effector.
        # The cylinder is used to have a HPPFCL shape at the end of the robot to make contact with the target

        # Obtaining the frame ID of the frame tool0
        ID_frame_tool0 = self.rmodel.getFrameId('tool0')
        # Obtaining the frame tool0
        frame_tool0 = self.rmodel.frames[ID_frame_tool0]
        # Obtaining the parent joint of the frame tool0
        parent_joint = frame_tool0.parentJoint
        # Obtaining the placement of the frame tool0
        Mf_endeff = frame_tool0.placement
        # Creating the endeff frame
        endeff_frame = pin.Frame("endeff", parent_joint, Mf_endeff, pin.BODY)
        _ = self.rmodel.addFrame(endeff_frame, False)

        # Creation of the geometrical model
        self.gmodel = self.robot.visual_model

        # Creation of the cylinder at the end of the end effector

        # Setting up the raddi of the cylinder
        endeff_radii, endeff_width = 1e-2, 1e-2
        # Creating a HPPFCL shape
        endeff_shape = hppfcl.Cylinder(endeff_radii, endeff_width)
        # Creating a pin.GeometryObject for the model of the robot
        geom_endeff = pin.GeometryObject(
            "endeff_geom", parent_joint, Mf_endeff, endeff_shape)
        geom_endeff.meshColor = self.color
        # Add the geometry object to the geometrical model
        self.gmodel.addGeometryObject(geom_endeff)

        if target:
            self._create_target()

        return self.robot, self.rmodel, self.gmodel

    def _create_target(self):
        """ Returns an updated version of the robot models with a sphere that can be used as a target.

        Returns
        -------
        robot
            Robot description of the said robot
        rmodel
            Model of the robot
        gmodel 
            Geometrical model of the robot
        """

        # Setup of the shape of the target (a sphere here)
        r_target = 5e-2*self.scale

        # Creation of the target

        # Creating the frame of the target

        M_target = self._generate_reachable_SE3_vector()

        target_frame = pin.Frame("target", self.rmodel.getJointId(
            "universe"), M_target, pin.BODY)
        target = self.rmodel.addFrame(target_frame, False)
        T_target = self.rmodel.frames[target].placement
        target_shape = hppfcl.Sphere(r_target)
        geom_target = pin.GeometryObject("target_geom", self.rmodel.getJointId(
            "universe"), T_target, target_shape)

        geom_target.meshColor = self.color
        self.gmodel.addGeometryObject(geom_target)

    def _generate_reachable_SE3_vector(self):
        """ Generate a SE3 vector that can be reached by the robot.

        Returns
        -------
        Reachable_SE3_vector
            SE3 Vector describing a reachable position by the robot
        """
        
        # Generate a random configuration of the robot, with consideration to its limits
        q = pin.randomConfiguration(self.rmodel)
        # Creation of a temporary model.Data, to have access to the forward kinematics.
        ndata = self.rmodel.createData()
        # Updating the model.Data with the framesForwardKinematics
        pin.framesForwardKinematics(self.rmodel, ndata, q)

        return ndata.oMf[self.rmodel.getFrameId('endeff')]


if __name__ == "__main__":

    robot_wrapper_test = robot_wrapper()
    robot, rmodel, gmodel = robot_wrapper_test(target=True)
    vis = create_visualizer(robot)
    print()
