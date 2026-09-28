
def build_covariance(sigma_x, sigma_y, sigma_yaw=1e-6):
    cov = [0.0] * 36
    cov[0] = float(sigma_x ** 2)
    cov[7] = float(sigma_y ** 2)
    cov[35] = float(sigma_yaw ** 2)
    return cov
