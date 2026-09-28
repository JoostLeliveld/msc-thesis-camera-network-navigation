"""Geometry validity is independent of statistical admission and camera order."""
from types import SimpleNamespace
from dataclasses import replace
import math
import numpy as np
import pytest
from reliability.contracts import CameraObservation
from unav_common.camera_model import ObliqueCameraModel
from reliability.projection import _floor_spd_2x2
from state.core.pixel_to_bev import PixelToBevTransformer


def camera():
    return ObliqueCameraModel(cam_pos=(-3.,-3.,5.),look_at=(0.,0.,0.),img_width=1280,img_height=720)


def observed(**changes):
    args=dict(camera_id='camera_B',calibration_id='warehouse_v2_camera_B',image_frame_id='camera_B',
              timestamp_s=1.,detection_valid=True,pixel_uv=(650.,420.),bbox_xyxy=(600.,380.,700.,420.),
              bbox_bottom_uv=(650.,420.),selected_pixel_source='bbox_bottom')
    args.update(changes)
    return CameraObservation(**args)


def validate(o):
    from reliability.observation_geometry import validate_observation_geometry
    return validate_observation_geometry(o,camera(),expected_camera_id='camera_B',
        expected_calibration_id='warehouse_v2_camera_B',expected_image_frame_id='camera_B')


def test_valid_boundary_and_miss():
    validate(observed())
    validate(observed(bbox_xyxy=(0.,0.,1280.,720.),bbox_bottom_uv=(640.,720.),pixel_uv=(640.,720.)))
    validate(observed(detection_valid=False,bbox_xyxy=None,pixel_uv=None,bbox_bottom_uv=None,selected_pixel_source='none'))


@pytest.mark.parametrize('changes',[
    {'calibration_id':'foreign'}, {'image_frame_id':'foreign'}, {'camera_id':'camera_A'},
    {'timestamp_s':-1.}, {'bbox_xyxy':(600.,380.,600.,420.)},
    {'bbox_xyxy':(700.,380.,600.,420.)}, {'bbox_xyxy':(-1.,380.,700.,420.)},
    {'bbox_xyxy':(600.,380.,1281.,420.)}, {'pixel_uv':(651.,420.)},
    {'bbox_bottom_uv':(650.,419.)}, {'selected_pixel_source':'mask_bottom'},
])
def test_semantic_mutations_rejected(changes):
    with pytest.raises(ValueError): validate(observed(**changes))


@pytest.mark.parametrize('cov', [[[1.,2.],[2.,1.]],[[1.,.1],[.2,1.]],[[1.,0.],[0.,float('nan')]]])
def test_validator_checks_covariance_even_without_contract_constructor(cov):
    values=vars(observed()).copy(); values['conditional_cov_uv']=cov
    with pytest.raises(ValueError): validate(SimpleNamespace(**values))


def test_original_image_dimensions_checked_when_supplied():
    values=vars(observed()).copy(); values.update(image_width=640,image_height=360)
    with pytest.raises(ValueError): validate(SimpleNamespace(**values))


@pytest.mark.parametrize('kwargs',[
    {'cam_pos':(0.,0.,0.),'look_at':(0.,0.,0.)},
    {'cam_pos':(0.,0.,5.),'look_at':(0.,0.,0.)},
    {'img_width':0}, {'img_width':1280.5}, {'fov_h_rad':math.pi},
    {'cam_pos':(math.nan,0.,5.)},
])
def test_degenerate_camera_rejected(kwargs):
    with pytest.raises(ValueError): ObliqueCameraModel(**kwargs)


def test_forward_floor_intersection_and_nan():
    c=camera(); a=c.H_inv
    v=-(a[2,0]*640+a[2,2])/a[2,1]
    assert c.pixel_to_world(640,v-10) is None
    assert c.pixel_to_world(640,v) is None
    assert c.pixel_to_world(float('nan'),400) is None
    assert c.pixel_to_world_at_z(640,400,float('nan')) is None


def test_projection_preserves_valid_homography_exactly():
    c=camera()
    for u,v in [(0.,0.),(1280.,720.),(640.,360.)]:
        w=c.H_inv@np.array([u,v,1.])
        assert c.pixel_to_world(u,v)==(w[0]/w[2],w[1]/w[2])


def test_unrepresentable_floor_is_explicit_refusal():
    with pytest.raises(ValueError): _floor_spd_2x2(((1e8,1e8),(1e8,1e8)),1e-12)


def test_pixel_covariance_full_matrix_against_independent_finite_difference():
    c=camera(); t=PixelToBevTransformer(c.cam_pos,c.look_at,1280,720,c.fov_h_rad)
    u,v=600.,450.; step=1e-3
    J=np.column_stack([(np.array(c.pixel_to_world(u+step*(i==0),v+step*(i==1)))-np.array(c.pixel_to_world(u-step*(i==0),v-step*(i==1))))/(2*step) for i in (0,1)])
    R=np.array([[4.,1.],[1.,2.]])
    np.testing.assert_allclose(t.pixel_covariance_to_metric(u,v,R),J@R@J.T,rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(t.pixel_noise_to_metric(u,v,.1),np.sqrt(np.diag(J@J.T))*.1,rtol=1e-8)


def test_covariance_mapping_does_not_sample_random_geometry():
    c=camera()
    rng=SimpleNamespace(normal=lambda *a,**kw:pytest.fail('covariance differentiation sampled random extrinsics'))
    t=PixelToBevTransformer(c.cam_pos,c.look_at,1280,720,c.fov_h_rad,rng=rng)
    R=t.pixel_covariance_to_metric(640.,400.,np.eye(2),transform_noise_sigma=.01)
    assert np.linalg.eigvalsh(R).min()>0


@pytest.mark.parametrize('pose',[
    '<pose>0 0 5 .1 .7 0</pose>',
    '<pose relative_to="other">0 0 5 0 .7 0</pose>',
    '<pose degrees="true">0 0 5 0 45 0</pose>',
    '<pose>0 0 nan 0 .7 0</pose>',
    '<pose>0 0 -5 0 .7 0</pose>',
])
def test_world_loader_refuses_unsupported_pose(tmp_path,pose):
    from reliability.projection import camera_model_from_world
    p=tmp_path/'world.sdf'
    p.write_text('<sdf><world name="w"><include><name>camera</name>'+pose+'</include></world></sdf>')
    with pytest.raises(RuntimeError): camera_model_from_world(p,include_name='camera')


def test_ambiguous_world_include_refused(tmp_path):
    from reliability.projection import camera_model_from_world
    p=tmp_path/'world.sdf'; entry='<include><name>camera</name><pose>0 0 5 0 .7 0</pose></include>'
    p.write_text('<sdf><world name="w">'+entry*2+'</world></sdf>')
    with pytest.raises(RuntimeError,match='Ambiguous'): camera_model_from_world(p,include_name='camera')


def test_all_frozen_raw_projections_unchanged():
    import csv,json
    from pathlib import Path
    from reliability.projection import camera_model_from_world
    root=Path(__file__).resolve().parents[2]
    capture=root/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
    if not capture.exists(): pytest.skip('frozen geometry fixture unavailable')
    manifest=json.loads((capture/'capture_manifest.json').read_text())
    world=root/'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
    models={c['camera_id']:camera_model_from_world(world,include_name=c['camera_model']) for c in manifest['cameras']}
    checked=0
    with (capture/'bias_update_interpretations.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['raw_valid']!='1': continue
            actual=models[row['camera_id']].pixel_to_world(float(row['u_bbox_bottom']),float(row['v_bbox_bottom']))
            assert actual==(float(row['raw_x']),float(row['raw_y']))
            checked+=1
    assert checked==6412


@pytest.mark.parametrize('bad',[np.array([[1.,2.],[2.,1.]]),np.array([[1.,.1],[.2,1.]]),np.eye(3)])
def test_optional_covariance_invalid_inputs_refused(bad):
    c=camera(); t=PixelToBevTransformer(c.cam_pos,c.look_at,1280,720,c.fov_h_rad)
    with pytest.raises(ValueError): t.pixel_covariance_to_metric(640.,400.,bad)


def test_strict_dimension_mode_refuses_missing_original_size():
    from reliability.observation_geometry import validate_observation_geometry
    kwargs=dict(expected_camera_id='camera_B',expected_calibration_id='warehouse_v2_camera_B',
                expected_image_frame_id='camera_B',require_image_dimensions=True)
    with pytest.raises(ValueError,match='image width'):
        validate_observation_geometry(observed(),camera(),**kwargs)
    values=vars(observed()).copy(); values.update(image_width=1280,image_height=720)
    validate_observation_geometry(SimpleNamespace(**values),camera(),**kwargs)


def test_unsupported_schema_refused():
    values=vars(observed()).copy(); values['schema_version']='foreign.v1'
    with pytest.raises(ValueError,match='schema'): validate(SimpleNamespace(**values))
