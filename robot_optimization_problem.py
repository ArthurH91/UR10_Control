import numpy as np
import pinocchio as pin
from robot_wrapper import robot_wrapper
from visualizer import create_visualizer
# import hppfcl

class robot_optimization():

    def __init__(self, rmodel: pin.Model,rdata: pin.Data, gmodel: pin.GeometryModel, gdata: pin.GeometryData, vis):
        """Initialize the class with the models and datas of the robot

        Parameters
        ----------
        rmodel : pin.Model
            Model of the robot
        rdata : pin.Data
            Data of the model of the robot
        gmodel : pin.GeometryModel
            Geometrical model of the robot
        gdata : pin.GeometryData
            Geometrical data of the model of the robot
        q : np.ndarray
            Array of configuration of the robot, size robot.nq
        vis : meschat visualizer
        """
        self.rmodel = rmodel
        self.rdata = rdata
        self.gmodel = gmodel
        self.gdata = gdata
        self.vis = vis


    def get_transform_from_Frame(self, frame_geom: str):
        """returns the transform given a frame name, takes the str name of the frame

        Parameters
        ----------
        frame_geom : str
            Name of the geometrical frame

        Returns
        -------
        SE3 Vector
            Transform of the given frame in SE3
        """
        frame_Id = self.gmodel.getGeometryId(frame_geom)
        return self.gdata.oMg[frame_Id]
    

    def _compute_distance_between_two_frames(self, frame_geom1: str, frame_geom2: str, q: np.ndarray):
        """ Computes the distance between 2 geometrical frames. They HAVE to be geometrical frames.

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
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.updateGeometryPlacements(self.rmodel, self.rdata, self.gmodel, self.gdata, q)

        # Getting the 2 poses of the frames 
        pose_frame1 = self.get_transform_from_Frame(frame_geom1)
        pose_frame2 = self.get_transform_from_Frame(frame_geom2)

        # Multiplying one frame by the inverse of the other to obtain the distance
        dist = pose_frame1 * pose_frame2.inverse()

        return np.linalg.norm(dist)
    
    def compute_cost_function(self, q: np.ndarray):
        """Computes the quadratic cost function, that is the squared distance between the target and the end-effector.

        Parameters
        ----------
        q : np.ndarray
            Array of the configuration of the robot

        Returns
        -------
        cost : float
            Square distance between the target and the end effector at the configuration q 
        """
        dist = self._compute_distance_between_two_frames("endeff_geom", "target_geom",q)
        return 0.5 * dist ** 2 
    


    def callback(self, q: np.ndarray):
        self.vis.display(q)

if __name__ == "__main__":
    
    robot_wrapper_test = robot_wrapper()
    robot, rmodel, gmodel = robot_wrapper_test(target=True)
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    vis = create_visualizer(robot)

    input()
    q = pin.randomConfiguration(rmodel)
    pin.framesForwardKinematics(rmodel, rdata, q)

    # THIS STEP IS MANDATORY OTHERWISE THE FRAMES AREN'T UPDATED
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata, q) 
    # vis.display(q)

    rob = robot_optimization(rmodel, rdata, gmodel, gdata, vis)
   