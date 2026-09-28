"""Execute the actual node callback without ROS; verify its published XY block."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from state.core.pixel_to_bev import PixelToBevTransformer
from state.core.noise import build_covariance


def message():
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=4,nanosec=0)),
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(),orientation=SimpleNamespace()),covariance=[]))


def callback():
    source=Path(__file__).resolve().parents[2]/'src/state/state/nodes/pixel_to_bev_state_node.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PixelToBevStateNode')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pixel_callback')
    scope=dict(math=math,np=np,PoseStamped=object,PoseWithCovarianceStamped=message,
        Float64MultiArray=SimpleNamespace,build_covariance=build_covariance,
        HEADING_SOURCE_CODES={'held_previous_heading':4.,'unknown':0.})
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    return scope['_pixel_callback']


def node(affine=None,sigma=.3):
    out=[]
    transformer=PixelToBevTransformer((-3.,-3.,5.),(0.,0.,0.),1280,720,1.5708)
    n=SimpleNamespace(pixel_noise_sigma=sigma,transform_noise_sigma=0.,_rng=SimpleNamespace(normal=lambda *a:0.),
        _transformer=transformer,_bev_affine=affine,bev_y_calibration_offset_m=.2,
        _matching_diag=lambda stamp:None,_last_yaw=.4,motion_yaw_sigma_rad=.2,odom_heading_sigma_rad=.1,
        yaw_noise_floor_rad=.01,use_odom_heading_fallback=False,infer_yaw_from_motion=False,
        _last_xy=None,_set_heading_source=lambda value:None,
        _yaw_to_quaternion=lambda yaw:(0.,0.,math.sin(yaw/2),math.cos(yaw/2)),
        frame_id='map_bev',_publisher=SimpleNamespace(publish=out.append),
        _heading_diag_pub=SimpleNamespace(publish=lambda value:None),
        _stamp_to_float=lambda stamp:4.,_odom_age_s=lambda stamp:math.nan)
    return n,out


def input_message():
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=4,nanosec=0)),
        pose=SimpleNamespace(position=SimpleNamespace(x=600.,y=450.)))


@pytest.mark.parametrize('affine',[None,(0.,-1.,2.,1.,0.,-3.),(1.2,.4,2.,-.3,.8,-3.)])
def test_full_covariance_and_affine_congruence(affine):
    n,out=node(affine); msg=input_message(); t=n._transformer
    raw=np.array(t.pixel_to_world(600.,450.)); eps=1e-3
    J=np.column_stack([(np.array(t.pixel_to_world(600.+eps*(i==0),450.+eps*(i==1)))-np.array(t.pixel_to_world(600.-eps*(i==0),450.-eps*(i==1))))/(2*eps) for i in (0,1)])
    R=J@J.T*n.pixel_noise_sigma**2
    A=np.eye(2) if affine is None else np.array([[affine[0],affine[1]],[affine[3],affine[4]]])
    offset=np.array([0.,.2]) if affine is None else np.array([affine[2],affine[5]])
    callback()(n,msg)
    assert len(out)==1
    published=out[0]; actual=np.array(published.pose.covariance).reshape(6,6)[:2,:2]
    np.testing.assert_allclose(actual,A@R@A.T,rtol=1e-8,atol=1e-12)
    assert abs(actual[0,1])>1e-8
    np.testing.assert_allclose([published.pose.pose.position.x,published.pose.pose.position.y],A@raw+offset)
    assert published.header.stamp is msg.header.stamp
    assert published.header.frame_id=='map_bev'
    assert published.pose.covariance[35]==pytest.approx(.2**2)


@pytest.mark.parametrize('R',[None,np.full((2,2),np.nan),np.array([[1.,2.],[2.,1.]]),np.array([[1.,.1],[.2,1.]])])
def test_invalid_covariance_has_no_publication_or_heading_commit(R):
    n,out=node(); n._transformer.pixel_covariance_to_metric=lambda *a,**kw:R
    callback()(n,input_message())
    assert out==[] and n._last_xy is None


def test_transform_only_covariance_is_propagated():
    n,out=node(sigma=0.); n.transform_noise_sigma=.01
    # Hold the sampled mean at nominal geometry; covariance remains a statement
    # about the declared transform uncertainty, not a random draw.
    n._transformer.rng=None
    callback()(n,input_message())
    R=np.array(out[0].pose.covariance).reshape(6,6)[:2,:2]
    assert np.linalg.eigvalsh(R).min()>0 and abs(R[0,1])>1e-10


def test_existing_precision_blend_is_preserved():
    n,out=node(); n._matching_diag=lambda stamp:{'yolo_detected_after_threshold':1.,'yolo_score_selected':.75}
    n.live_detection_trust=lambda **kw:.75
    n.R_visible_std=2.5; n.R_update_miss_std=40.
    callback()(n,input_message())
    t=n._transformer; eps=1e-3
    J=np.column_stack([(np.array(t.pixel_to_world(600.+eps*(i==0),450.+eps*(i==1)))-np.array(t.pixel_to_world(600.-eps*(i==0),450.-eps*(i==1))))/(2*eps) for i in (0,1)])
    expected=J@J.T/(.75/2.5**2+.25/40.**2)
    np.testing.assert_allclose(np.array(out[0].pose.covariance).reshape(6,6)[:2,:2],expected,rtol=1e-8,atol=1e-12)
