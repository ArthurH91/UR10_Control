import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat


def create_visualizer(robot, grid=False, axes=False):
    """ Create a visualizer using meshcat, allowing a robot to be visualized. 

    Parameters
    ----------
    robot : 
        Robot unwrapped from pinocchio
    grid : bool, optional
        Whether the user wants the grid, by default False
    axes : bool, optional
        Whether the user wants the axes, by default False

    Returns
    -------
    vis
        A vis object, can be updated with vis.dispay(q), where q is an np.ndarray of robot.nq dimensions.
    """
    Viewer = pin.visualize.MeshcatVisualizer
    vis = Viewer(robot.model, robot.collision_model, robot.visual_model)
    vis.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
    vis.viewer.delete()
    vis.loadViewerModel()
    if not grid:
        vis.viewer["/Grid"].set_property("visible", False)
    if not axes:
        vis.viewer["/Axes"].set_property("visible", False)
    return vis

