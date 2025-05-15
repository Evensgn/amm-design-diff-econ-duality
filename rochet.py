import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
import os
from pathlib import Path
import scipy.stats as stats


class RochetNet(nn.Module):
    def __init__(self, num_items, num_menus, alloc_mag=1.0, demand_type="additive"):
        super(RochetNet, self).__init__()

        self.alloc_mag = alloc_mag  # range is [-alloc_mag, +alloc_mag]
        self.softmax_temp = 100.0

        self.demand_type = demand_type

        # trainable parameter matrix of size num_menus x num_items
        # extra "dummy" param
        self.alloc_param = nn.Parameter(torch.rand(num_menus, num_items + 1))

        # trainable parameter vector of size num_menus
        self.payments = nn.Parameter(torch.rand(num_menus))
        self.num_items = num_items
        self.num_menus = num_menus

    def menu_and_payment(self):
        if self.demand_type == "additive":
            allocs = 2.0 * self.alloc_mag * F.sigmoid(self.alloc_param) - self.alloc_mag
        elif self.demand_type == "unit":
            allocs = (
                2.0 * self.alloc_mag * F.softmax(self.alloc_param, dim=-1)
                - self.alloc_mag
            )

        # for unit demand, the dummy param serves a purpose to allow a <=1 constraint
        allocs = allocs[..., :-1]
        allocs = torch.cat(
            [
                allocs,
                torch.zeros(1, allocs.shape[1], device=next(self.parameters()).device),
            ],
            dim=0,
        )

        return allocs, torch.cat(
            [self.payments, torch.zeros(1, device=next(self.parameters()).device)]
        )

    def forward(self, x, train=True):
        allocs, payments = self.menu_and_payment()

        # calculate the expected utility of each menu
        expected_utility = (
            torch.sum(allocs[None, ...] * x[..., None, :], dim=-1) - payments[None, ...]
        )

        if train:
            util_weights = F.softmax(expected_utility * self.softmax_temp, dim=-1)

            # chosen_alloc shape: [..., num_items]
            chosen_alloc = torch.sum(
                util_weights[..., :, None] * allocs[None, ...], dim=-2
            )

            # chosen_payment shape: [...]
            chosen_payment = torch.sum(util_weights * payments[None, ...], dim=-1)

        else:
            chosen_ind = torch.argmax(expected_utility, dim=-1)
            chosen_alloc = torch.index_select(allocs, 0, chosen_ind.flatten()).reshape(
                *chosen_ind.shape, -1
            )
            chosen_payment = torch.index_select(
                payments, 0, chosen_ind.flatten()
            ).reshape(*chosen_ind.shape)

        return chosen_alloc, chosen_payment


def visualize_model_output(model, resolution=100, item=0):
    x_range = np.linspace(0, 1, resolution)
    y_range = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.stack([xx, yy], axis=-1)

    with torch.no_grad():
        grid_points = torch.tensor(
            grid_points, dtype=torch.float32, device=model.alloc_param.device
        )
        allocs, payments = model(grid_points, train=False)

    item_allocs = allocs[..., item]
    heatmap = item_allocs.cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(ax.images[0])

    return fig


def noise_trading_loss(allocs, payments, c, x):
    # calculates loss for each of a batch of inputs
    return payments - torch.sum(c[None, :] * allocs, dim=-1)


def construct_linear_loss(l=0.1):
    def linear_loss(allocs, payments, c, x):
        new_c = l * x + (1 - l) * c
        return payments - torch.sum(new_c[None, :] * allocs, dim=-1)

    return linear_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and visualize the RochetNet model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--num_menus", type=int, default=1024, help="Number of menus.")
    parser.add_argument("--num_items", type=int, default=2, help="Number of items.")
    parser.add_argument(
        "--num_iters", type=int, default=500, help="Number of iterations."
    )
    parser.add_argument(
        "--resolution", type=int, default=100, help="Resolution for visualization."
    )
    parser.add_argument(
        "--loss_lambda", type=float, default=0.0, help="lambda for linear loss"
    )

    parser.add_argument(
        "--initial_c",
        help="comma-separated array of initial c values for all goods",
        type=str,
        default="0.0",
    )

    parser.add_argument(
        "--l2_reg", type=float, default=0.0, help="l2 regularization strength"
    )

    parser.add_argument(
        "--distribution", type=str, default="uniform", help="distribution type"
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="lr")

    args = parser.parse_args()

    # set random seed
    wandb.init(project="rochet-net-training", config=vars(args))

    scratch_plots = "scratch_plots_" + wandb.run.id
    os.makedirs(scratch_plots, exist_ok=True)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    # Check if MPS is available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    loss_fn = construct_linear_loss(args.loss_lambda)

    init_c_args = [float(x) for x in args.initial_c.split(",") if x]
    if len(init_c_args) == 1:
        c = (torch.ones(args.num_items) * init_c_args[0]).to(device)
        print("Using constant c: {}".format(c))
    elif len(init_c_args) == args.num_items:
        c = torch.tensor(init_c_args).to(device)
    else:
        raise ValueError(
            "initial_c must be either a single value or a comma-separated list of length num_items"
        )

    rochet_net = RochetNet(args.num_items, args.num_menus, alloc_mag=1.0).to(device)
    wandb.watch(rochet_net)  # Watch the model to log its gradients and parameters

    optimizer = optim.Adam(rochet_net.parameters(), lr=args.lr)

    if args.distribution == "uniform":
        dist_generator = lambda size_arg: torch.rand(
            size_arg, args.num_items, device=device
        )

    elif "linear_corr_uniform" in args.distribution:
        # Linear correlated uniform distribution: Y = alpha * X + (1 - alpha ) * U
        def correlated_uniform(size_arg, alpha):
            # alpha value range: [0, 1]
            X = torch.rand(size_arg, 1, device=device)  # X ~ U(0,1)
            U = torch.rand(size_arg, 1, device=device)  # U ~ U(0,1)
            Y = alpha * X + (1 - alpha) * U  # Correlated Y
            return torch.cat([X, Y], dim=1)

        alpha_value = float(args.distribution.split("_")[3])
        dist_generator = lambda size_arg: correlated_uniform(size_arg, alpha_value)

    elif "gaussian_copula" in args.distribution:
        # Samples in [0,1]^2 space with Gaussian copula correlation
        def gaussian_copula(size_arg, rho):
            # rho value range: [-1, 1]
            mean = torch.tensor([0.0, 0.0], device=device)
            # Correlation matrix
            cov = torch.tensor([[1.0, rho], [rho, 1.0]], device=device)

            # Sample from multivariate normal
            mvn_samples = torch.distributions.MultivariateNormal(
                mean, covariance_matrix=cov
            ).sample((size_arg,))

            # Transform to uniform via Gaussian CDF
            uniform_samples = torch.tensor(
                stats.norm.cdf(mvn_samples.cpu().numpy()),
                device=device,
                dtype=torch.float32,
            )

            return uniform_samples

        rho_value = float(args.distribution.split("_")[2])
        dist_generator = lambda size_arg: gaussian_copula(size_arg, rho_value)

    elif "smallbox" in args.distribution:
        small_box_size = float(args.distribution.split("_")[1])
        small_box_base = np.array(
            [float(x) for x in args.distribution.split("_")[2].split(",")]
        )
        dist_generator = lambda size_arg: small_box_size * torch.rand(
            size_arg, args.num_items, device=device
        ) + torch.tensor(small_box_base, device=device, dtype=torch.float32)
    elif args.distribution == "beta":
        # default beta 1,2
        dist_generator = (
            lambda size_arg: torch.distributions.Beta(2, 2)
            .sample((size_arg, args.num_items))
            .detach()
            .to(device)
        )
    elif args.distribution == "truncnorm":
        dist_generator = lambda size_arg: torch.clamp(
            (1.0 / 8.0) * torch.randn(size_arg, args.num_items, device=device) + 0.5,
            min=0.0,
            max=1.0,
        )
    else:
        raise ValueError("distribution not implemented")

    for n in range(args.num_iters):
        x = dist_generator(args.batch_size)
        optimizer.zero_grad()
        allocs, payments = rochet_net(x, train=True)
        # loss is - (payments - c*allocs)
        loss = -torch.mean(loss_fn(allocs, payments, c, x))

        # l2 regularization
        # loss += args.l2_reg * torch.mean(rochet_net.alloc_param**2)
        loss.backward()
        optimizer.step()
        # Logging training loss
        wandb.log({"training_loss": loss.item()})

        if (n % 100) == 0:
            with torch.no_grad():
                x = dist_generator(args.batch_size)
                allocs, payments = rochet_net(x, train=False)
                eval_loss = -torch.mean(loss_fn(allocs, payments, c, x))
                if args.num_items == 2:
                    item1_fig = visualize_model_output(
                        rochet_net, resolution=args.resolution, item=0
                    )
                    item2_fig = visualize_model_output(
                        rochet_net, resolution=args.resolution, item=1
                    )
            if args.num_items == 2:
                item1_path = Path(scratch_plots) / Path(f"item1_eval_{n}.png")
                item2_path = Path(scratch_plots) / Path(f"item2_eval_{n}.png")
                item1_fig.savefig(item1_path)
                item2_fig.savefig(item2_path)
                plt.close()
                plt.close()

                wandb.log(
                    {
                        "eval_loss": eval_loss.item(),
                        "model_state": rochet_net.state_dict(),
                        "item1_plot_eval": wandb.Image(str(item1_path)),
                        "item2_plot_eval": wandb.Image(str(item2_path)),
                    }
                )
            else:
                wandb.log(
                    {
                        "eval_loss": eval_loss.item(),
                        "model_state": rochet_net.state_dict(),
                    }
                )

            print("eval loss (profit): {}".format(eval_loss.item()))

    if args.num_items == 2:
        item1_fig = visualize_model_output(
            rochet_net, resolution=args.resolution, item=0
        )
        item2_fig = visualize_model_output(
            rochet_net, resolution=args.resolution, item=1
        )

        # Save the images and model parameters locally and to wandb
        item1_path_final = Path(scratch_plots) / Path(f"item1_final.png")
        item2_path_final = Path(scratch_plots) / Path(f"item2_final.png")
        item1_fig.savefig(item1_path_final)
        item2_fig.savefig(item2_path_final)
        wandb.log(
            {
                "item1_plot_final": wandb.Image(str(item1_path_final)),
                "item2_plot_final": wandb.Image(str(item2_path_final)),
                "final_model_state": rochet_net.state_dict(),
            }
        )
    else:
        wandb.log({"final_model_state": rochet_net.state_dict()})

    torch.save(rochet_net.cpu().state_dict(), Path("model_weights.pth"))
    wandb.save(str(Path("model_weights.pth")))

    # Finish the run
    wandb.finish()
