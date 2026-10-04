import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import torch
    import gpytorch

    import pandas as pd
    import numpy as np

    from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
    from gpytorch.priors import HalfNormalPrior

    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.float64)


    class GroupKernel(Kernel):
        """k(g, g') = 1 if same group else 0. Wrap in ScaleKernel to get tau^2."""

        def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
            if diag:
                return (x1 == x2).all(-1).to(x1.dtype)
            return (x1.unsqueeze(-2) == x2.unsqueeze(-3)).all(-1).to(x1.dtype)


    class ExpGP(gpytorch.models.ExactGP):
        """y = m + f(x) + b_g + eps, with inputs X = [x, group_index] (two columns)."""

        def __init__(self, X, y, likelihood, tau_scale):
            super().__init__(X, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.k_f = ScaleKernel(RBFKernel(active_dims=[0]))            # sigma_f^2 * SE(x, x'; ell)
            self.k_g = ScaleKernel(GroupKernel(active_dims=[1]),          # tau^2 * delta_gg'
                                   outputscale_prior=HalfNormalPrior(tau_scale))

        def forward(self, X):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(X), self.k_f(X) + self.k_g(X)
            )


    def fit_exp_covariance(x, g, y, x_star, n_iter=300, lr=0.05):
        """
        x, g, y : (N,) raw x, integer group index, response
        x_star  : (M,) design points
        Returns (mean, Sigma_exp): posterior mean and covariance of the population
        curve f at x_star (excludes group offsets b_g and observation noise).
        """
        x, y, x_star = (torch.as_tensor(a, dtype=torch.float64) for a in (x, y, x_star))
        g = torch.as_tensor(g, dtype=torch.float64)

        y_mu, y_sd = y.mean(), y.std()
        ys = (y - y_mu) / y_sd                                    # standardise y
        X = torch.stack([x, g], dim=-1)
        Xs = torch.stack([x_star, torch.zeros_like(x_star)], dim=-1)   # group column is ignored by k_f

        # prior scale for tau: spread of group means (in standardised units)
        tau_scale = max(float(torch.stack([ys[g == k].mean() for k in g.unique()]).std()), 0.1)

        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = ExpGP(X, ys, lik, tau_scale)
        model.k_f.base_kernel.lengthscale = 0.2 * float(x.max() - x.min())

        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        for _ in range(n_iter):
            opt.zero_grad()
            loss = -mll(model(X), ys)
            loss.backward()
            opt.step()

        # Posterior of f only: K_f(*,*) - K_f(*,X) K_yy^{-1} K_f(X,*)
        model.eval(); lik.eval()
        with torch.no_grad():
            Kyy = (model.k_f(X) + model.k_g(X)).to_dense() + lik.noise * torch.eye(len(x))
            Ks = model.k_f(Xs, X).to_dense()
            Kss = model.k_f(Xs, Xs).to_dense()
            m = model.mean_module.constant
            mean = m + Ks @ torch.linalg.solve(Kyy, ys - m)
            Sigma = Kss - Ks @ torch.linalg.solve(Kyy, Ks.T)

        Sigma = 0.5 * (Sigma + Sigma.T)                           # symmetrise
        mean = (mean * y_sd + y_mu).numpy()
        Sigma = (Sigma * y_sd**2).numpy()                         # back to original units
        ell = model.k_f.base_kernel.lengthscale.item()

        print(f"sigma_f={model.k_f.outputscale.sqrt().item():.3f}  "
              f"ell={model.k_f.base_kernel.lengthscale.item():.3f}  "
              f"tau={model.k_g.outputscale.sqrt().item():.3f}  "
              f"sigma_n={lik.noise.sqrt().item():.3f}  (standardised units)")

        return mean, Sigma, ell
    return fit_exp_covariance, np, os, pd, plt


@app.cell
def _(pd):
    site_dataframe = pd.read_csv("wetlab_data/site_dataframe.csv")
    return (site_dataframe,)


@app.cell
def _(site_dataframe):
    site_dataframe
    return


@app.cell
def _(os):
    save_directory = os.path.join("wetlab_data", "gp_results")
    return (save_directory,)


@app.cell
def _(fit_exp_covariance, np, os, save_directory, site_dataframe):
    QUERY_COUNTS = np.linspace(50, 300, 11).astype(int)
    METRICS = ["mean_speed", "mean_mr", "anni", "coherency_fraction"]

    def extract_fit_values(phenotype):
        # Get relevant datasets:
        phenotype_mask = site_dataframe["phenotype"] == phenotype
        cell_count = site_dataframe.loc[phenotype_mask, "particle_count"].to_numpy()
        _, group_var = np.unique(site_dataframe.loc[phenotype_mask, "experiment"], return_inverse=True)

        for metric_label in METRICS:
            # Get output:
            metric_values = site_dataframe.loc[phenotype_mask, metric_label].to_numpy()
            mean, sigma, ell = fit_exp_covariance(cell_count, group_var, metric_values, QUERY_COUNTS, n_iter=300, lr=0.05)

            # Save outputs:
            np.save(os.path.join(save_directory, f"{phenotype}_{metric_label}_mean.npy"), mean)
            np.save(os.path.join(save_directory, f"{phenotype}_{metric_label}_sigma.npy"), sigma)
            np.save(os.path.join(save_directory, f"{phenotype}_{metric_label}_lengthscale.npy"), ell)
    return QUERY_COUNTS, extract_fit_values


@app.cell
def _(extract_fit_values):
    extract_fit_values("CTL")
    return


@app.cell
def _(extract_fit_values):
    extract_fit_values("RD")
    return


@app.cell
def _(QUERY_COUNTS, np, os, plt, save_directory):
    def plot_gp_regression(metric_label):
        fig, ax = plt.subplots()

        for phenotype in ["CTL", "RD"]:

            mean = np.load(os.path.join(save_directory, f"{phenotype}_{metric_label}_mean.npy"))
            sigma = np.load(os.path.join(save_directory, f"{phenotype}_{metric_label}_sigma.npy"))
            se = np.sqrt(np.diag(sigma))

            # Plot regression:
            ax.plot(QUERY_COUNTS, mean)
            ax.fill_between(QUERY_COUNTS, mean - se, mean + se, alpha=0.2)

        plt.show()

    plot_gp_regression("coherency_fraction")
    return


@app.cell
def _(np, os, plt, save_directory):
    def plot_gp_sigma(metric_label):
        fig, axs = plt.subplots(1, 2)

        count = 0
        for phenotype in ["CTL", "RD"]:
            sigma = np.load(os.path.join(save_directory, f"{phenotype}_{metric_label}_sigma.npy"))
            axs[count].imshow(sigma / sigma.max(), vmin=0.1, vmax=1.0)
            count += 1

        plt.show()

    plot_gp_sigma("anni")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
