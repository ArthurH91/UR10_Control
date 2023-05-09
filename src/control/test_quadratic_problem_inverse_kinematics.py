import unittest

import numpy as np
import pinocchio as pin
import hppfcl

from quadratic_problem_inverse_kinematics import QuadratricProblemInverseKinematics
from wrapper_robot import RobotWrapper
from utils import numdiff, generate_reachable_target

np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)


class TestQuadraticProblemNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup of the environement
        robot_wrapper = RobotWrapper()
        cls._robot, cls._rmodel, cls._gmodel = robot_wrapper()
        cls._rdata = cls._rmodel.createData()
        cls._gdata = cls._gmodel.createData()

        # Configuration array reaching the target


        cls._p_target, cls._q_target = generate_reachable_target(
            cls._rmodel, cls._rdata, returnConfiguration=True
        )
        cls._q_init = pin.randomConfiguration(cls._rmodel)

        cls._TARGET_SHAPE = hppfcl.Sphere(5e-2)

        # Configuring the computation of the QP
        target = cls._p_target.translation
        cls._QP = QuadratricProblemInverseKinematics(
            cls._rmodel,
            cls._rdata,
            cls._gmodel,
            cls._gdata,
            cls._p_target,
            cls._TARGET_SHAPE
        )


    # Tests
    def test_gradient_finite_difference(self):
        """Testing the gradient of the cost with the finite difference method"""

        grad_numdiff = numdiff(self._QP.cost, self._q_init)
        gradval = self._QP.grad(self._q_init)
        self.assertAlmostEqual(
            np.linalg.norm(gradval - grad_numdiff),
            0,
            places=5,
            msg="The gradient is not the same as the finite difference one",
        )


if __name__ == "__main__":
    # Start of the unit tests
    unittest.main()
