import numpy as np
from planning.core import belief_correction as bc


def test_scaled_gain_covariance_matches_independent_joseph_formula():
    rng=np.random.default_rng(91)
    for scale in (0., .1, .5, 1.):
        A=rng.normal(size=(3,3));P=A@A.T+np.eye(3)*.01
        H=np.array([[1.,0.,.2],[0.,1.,-.1]])
        R=np.array([[.2,.03],[.03,.1]])
        S=H@P@H.T+R; G=P@H.T
        lin=bc.Linearization(z=np.ones(2),mu_y=np.zeros(2),Gamma=G,Sigma_y=S,
            R_eff=R,S_eff=P,gain_scale=scale)
        got=bc.compute_update(np.zeros(3),lin,cov_eig_floor=1e-12)
        K=scale*np.linalg.solve(S,G.T).T
        J=np.eye(3)-K@H
        np.testing.assert_allclose(got.next_S,J@P@J.T+K@R@K.T,atol=1e-12)
        assert np.linalg.eigvalsh(got.next_S).min()>0
