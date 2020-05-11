import pybullet as pb
import numpy as np
import pybullet_data

physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF(‘plane.urdf’)