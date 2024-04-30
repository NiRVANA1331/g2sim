from gaussian_regression_1D import *


def main():
    
    gt_mu = 5
    gt_sigma = 3
    mu_start = 6.0
    sigma_start = 2.0

    env = Env(gt_mu,gt_sigma)
    train(env, 20000, mu_start, sigma_start)


if __name__ == "__main__":
    main()
