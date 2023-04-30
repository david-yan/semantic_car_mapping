import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_cuboid(cuboid, q):
    c_c = np.array([cuboid['x'], cuboid['y'], cuboid['z']])
    q_c = R.from_quat([cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w']])
    c_tf = q.apply(c_c)
    print('center before:', c_c)
    print('center after:', c_tf)
    q_tf = (q * q_c).as_quat()

    tf_cuboid = {
        'x': c_tf[0],
        'y': c_tf[1],
        'z': c_tf[2],
        'qx': q_tf[0],
        'qy': q_tf[1],
        'qz': q_tf[2],
        'w': q_tf[3]
    }
    return tf_cuboid