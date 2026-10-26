"""Unit tests for release acceptance, independent of renderer or MuJoCo."""
import unittest
from types import SimpleNamespace
import numpy as np
from hang_part import HangingSession

class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.arm=object.__new__(HangingSession)
        self.arm.ids={'part':(0,set())}
        self.arm.sim=SimpleNamespace(data=SimpleNamespace(body_xmat=np.array([np.eye(3).ravel()])))
        self.goal=np.array([-.4,.39,1.67])
        self.rows=[dict(target_cap_m=self.goal.tolist(),target_mass_kg=1.,target_linear_speed_m_s=0.,target_angular_speed_rad_s=0.,contacts=[dict(
            other='hook_shell1_1_cell_12_collision_006',object_geom='part_cap_collision',distance_m=-.00001,
            force_N=9.81,vertical_force_N=9.81,normal_vertical=1.)]) for _ in range(100)]
    def verify(self):return self.arm.verify_hanging('part',1,self.goal,self.rows)['success']
    def test_cap_supported_by_correct_layer(self):self.assertTrue(self.verify())
    def test_stem_snag_rejected(self):
        self.rows[0]['contacts'][0]['object_geom']='part_beam_collision_00'
        self.assertFalse(self.verify())
    def test_wrong_layer_rejected(self):
        self.rows[0]['contacts'][0]['other']='hook_shell2_1_cell_12_collision_006'
        self.assertFalse(self.verify())
    def test_finger_support_rejected(self):
        self.rows[0]['contacts'][0]['other']='gripper0_finger1_pad_collision'
        self.assertFalse(self.verify())
    def test_unsupported_rejected(self):
        self.rows[0]['contacts']=[];self.assertFalse(self.verify())
    def test_penetration_rejected(self):
        self.rows[0]['contacts'][0]['distance_m']=-.002;self.assertFalse(self.verify())
    def test_drift_rejected(self):
        self.rows[0]['target_cap_m']=(self.goal+[.01,0,0]).tolist();self.assertFalse(self.verify())
    def test_fast_motion_rejected(self):
        self.rows[0]['target_linear_speed_m_s']=.05;self.assertFalse(self.verify())
    def test_light_lateral_neck_touch_allowed(self):
        self.rows[0]['contacts'].append(dict(other='hook_shell1_1_cell_12_collision_006',object_geom='part_beam_collision_00',distance_m=-.000001,force_N=.1,vertical_force_N=.001,normal_vertical=0.))
        self.assertTrue(self.verify())
    def test_neck_load_rejected(self):
        self.rows[0]['contacts'].append(dict(other='hook_shell1_1_cell_12_collision_006',object_geom='part_beam_collision_00',distance_m=-.000001,force_N=1.,vertical_force_N=1.,normal_vertical=1.))
        self.assertFalse(self.verify())
    def test_insufficient_cap_load_rejected(self):
        self.rows[0]['contacts'][0]['vertical_force_N']=7.0
        self.assertFalse(self.verify())
    def test_side_of_cap_is_not_bearing_surface(self):
        self.rows[0]['contacts'][0]['normal_vertical']=0.
        self.assertFalse(self.verify())
    def test_online_corrections_are_bounded_rigid_poses(self):
        from scipy.spatial.transform import Rotation
        variants=list(HangingSession.placement_variants(np.zeros(3),np.eye(3)))
        self.assertGreater(len(variants),1)
        for position,rotation,_ in variants:
            self.assertLessEqual(np.linalg.norm(position),.00801)
            self.assertTrue(np.allclose(rotation.T@rotation,np.eye(3)))
            self.assertLessEqual(Rotation.from_matrix(rotation).magnitude(),np.deg2rad(10.001))
    def test_plate_top_detected_without_counting_teeth_or_rear_wall(self):
        from face_rack import count_head_depth_points
        cloud=np.array([[0,0,0],[.01,.01,0],[-.01,-.01,0],[0,0,-.002],[0,0,-.001],[0,.024,0]])
        self.assertEqual(count_head_depth_points(cloud,np.zeros(3),.075),3)

if __name__=='__main__':unittest.main()
