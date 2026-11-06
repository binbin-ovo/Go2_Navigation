"""IDL for ``rt/utlidar/imu``. The installed SDK lacks sensor_msgs Imu_; layout is standard."""

from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds import idl
from cyclonedds.idl import types
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Vector3_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Imu_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
    header: Header_
    orientation: Quaternion_
    orientation_covariance: types.array[types.float64, 9]
    angular_velocity: Vector3_
    angular_velocity_covariance: types.array[types.float64, 9]
    linear_acceleration: Vector3_
    linear_acceleration_covariance: types.array[types.float64, 9]
