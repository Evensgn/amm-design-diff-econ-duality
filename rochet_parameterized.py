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


class RochetNet(nn.Module):
    def __init__(
        self,
        num_items,
        num_menus,
        hidden_layer_sizes,
        alloc_mag=1.0,
        demand_type="additive",
        softmax_temp=100.0,
    ):
        super(RochetNet, self).__init__()

        self.alloc_mag = alloc_mag  # range is [-alloc_mag, +alloc_mag]
        self.softmax_temp = softmax_temp

        self.demand_type = demand_type

        self.num_items = num_items
        self.num_menus = num_menus

        # Input: (c, l)
        # input size = num_items + 1
        # Output: allocations and payments for each menu item
        # output size = num_menus * (num_items + 1) + num_menus
        layer_sizes = (
            [num_items + 1]
            + hidden_layer_sizes
            + [num_menus * (num_items + 1) + num_menus]
        )

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation on the last layer
                layers.append(nn.ReLU())
        self.menu_model = nn.Sequential(*layers)

    def menu_and_payment(self, c, l):
        batch_size = c.shape[0]
        menu_param = self.menu_model(torch.cat([c, l], dim=-1))
        alloc_param = menu_param[..., : -self.num_menus].reshape(
            batch_size, -1, self.num_items + 1
        )
        payments = menu_param[..., -self.num_menus :]

        if self.demand_type == "additive":
            allocs = 2.0 * self.alloc_mag * F.sigmoid(alloc_param) - self.alloc_mag
        elif self.demand_type == "unit":
            allocs = (
                2.0 * self.alloc_mag * F.softmax(alloc_param, dim=-1) - self.alloc_mag
            )

        # for unit demand, the dummy param serves a purpose to allow a <=1 constraint
        allocs = allocs[..., :-1]
        allocs = torch.cat(
            [
                allocs,
                torch.zeros(
                    batch_size, 1, self.num_items, device=next(self.parameters()).device
                ),
            ],
            dim=1,
        )

        payments = torch.cat(
            [
                payments,
                torch.zeros(batch_size, 1, device=next(self.parameters()).device),
            ],
            dim=1,
        )
        return allocs, payments

    def forward(self, x, c, l, train=True):
        batch_size = x.shape[0]

        allocs, payments = self.menu_and_payment(c, l)

        # calculate the expected utility of each menu
        expected_utility = torch.einsum("bmn,bn->bm", allocs, x) - payments

        if train:
            util_weights = F.softmax(expected_utility * self.softmax_temp, dim=-1)

            # chosen_alloc shape: [batch_size, num_items]
            chosen_alloc = torch.einsum("bm,bmn->bn", util_weights, allocs)

            # chosen_payment shape: [batch_size]
            chosen_payment = torch.einsum("bm,bm->b", util_weights, payments)

        else:
            chosen_ind = torch.argmax(expected_utility, dim=-1)

            # chosen_alloc shape: [batch_size, num_items]
            chosen_alloc = allocs[torch.arange(batch_size), chosen_ind]

            # chosen_payment shape: [batch_size]
            chosen_payment = payments[torch.arange(batch_size), chosen_ind]

        return chosen_alloc, chosen_payment


def visualize_model_output(model, resolution=100, item=0, c=(0.5, 0.5), l=0.0):
    x_range = np.linspace(0, 1, resolution)
    y_range = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    device = model.menu_model.parameters().__next__().device
    c_tensor = torch.tensor(
        [c] * grid_points.shape[0],
        dtype=torch.float32,
        device=device,
    )
    l_tensor = torch.ones(grid_points.shape[0], 1, device=device) * l

    with torch.no_grad():
        grid_points = torch.tensor(
            grid_points,
            dtype=torch.float32,
            device=device,
        )
        allocs, payments = model(grid_points, c_tensor, l_tensor, train=False)

    allocs = allocs.reshape(resolution, resolution, -1)
    item_allocs = allocs[..., item]
    heatmap = item_allocs.cpu().numpy()

    fig_c_12_l_0, ax = plt.subplots()
    ax.imshow(heatmap, cmap="RdBu_r", vmin=-1, vmax=1)
    fig_c_12_l_0.colorbar(ax.images[0])

    return fig_c_12_l_0


def noise_trading_loss(allocs, payments, c, l, x):
    # calculates loss for each of a batch of inputs
    return payments - torch.einsum("bn,bn->b", allocs, c)


def linear_loss(allocs, payments, c, l, x):
    new_c = l * x + (1 - l) * c
    return payments - torch.einsum("bn,bn->b", allocs, new_c)


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
        "--hidden_layer_sizes",
        type=str,
        default="32,32",  # Example default
        help="Comma-separated list of hidden layer sizes (e.g., '32,32')",
    )

    parser.add_argument(
        "--num_iters", type=int, default=500, help="Number of iterations."
    )
    parser.add_argument(
        "--resolution", type=int, default=100, help="Resolution for visualization."
    )

    parser.add_argument(
        "--l2_reg", type=float, default=0.0, help="l2 regularization strength"
    )

    parser.add_argument(
        "--distribution", type=str, default="uniform", help="distribution type"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")

    parser.add_argument(
        "--softmax_temp", type=float, default=100.0, help="softmax temperature"
    )

    parser.add_argument("--note", type=str, default="", help="note for the run")

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

    loss_fn = linear_loss

    hidden_layer_sizes = [int(size) for size in args.hidden_layer_sizes.split(",")]
    rochet_net = RochetNet(
        args.num_items,
        args.num_menus,
        hidden_layer_sizes,
        alloc_mag=1.0,
        softmax_temp=args.softmax_temp,
    ).to(device)

    wandb.watch(rochet_net)  # Watch the model to log its gradients and parameters

    optimizer = optim.Adam(rochet_net.parameters(), lr=args.lr)

    if args.distribution == "uniform":
        type_dist_generator = lambda size_arg: torch.rand(
            size_arg, args.num_items, device=device
        )
    elif args.distribution == "beta":
        # default beta 1,2
        type_dist_generator = (
            lambda size_arg: torch.distributions.Beta(2, 2)
            .sample((size_arg, args.num_items))
            .detach()
            .to(device)
        )
    elif args.distribution == "truncnorm":
        type_dist_generator = lambda size_arg: torch.clamp(
            (1.0 / 8.0) * torch.randn(size_arg, args.num_items, device=device) + 0.5,
            min=0.0,
            max=1.0,
        )
    else:
        raise ValueError("distribution not implemented")

    # The c distribution is uniform
    c_dist_generator = lambda size_arg: torch.rand(
        size_arg, args.num_items, device=device
    )

    # The l distribution is uniform
    l_dist_generator = lambda size_arg: torch.rand(size_arg, 1, device=device)

    for n in range(args.num_iters):
        x = type_dist_generator(args.batch_size)
        c = c_dist_generator(args.batch_size)
        l = l_dist_generator(args.batch_size)

        optimizer.zero_grad()
        allocs, payments = rochet_net(x, c, l, train=True)
        # loss is - (payments - c*allocs)
        loss = -torch.mean(loss_fn(allocs, payments, c, l, x))

        # l2 regularization
        # loss += args.l2_reg * torch.mean(rochet_net.alloc_param**2)
        loss.backward()
        optimizer.step()
        # Logging training loss
        wandb.log({"training_loss": loss.item()})

        if (n % 100) == 0:
            with torch.no_grad():
                x = type_dist_generator(args.batch_size)
                c = c_dist_generator(args.batch_size)
                l = l_dist_generator(args.batch_size)
                allocs, payments = rochet_net(x, c, l, train=False)
                eval_loss = -torch.mean(loss_fn(allocs, payments, c, l, x))
                if args.num_items == 2:
                    fig_c_center_l_0_item_0 = visualize_model_output(
                        rochet_net,
                        resolution=args.resolution,
                        item=0,
                        c=(0.5, 0.5),
                        l=0.0,
                    )
                    fig_c_center_l_0_item_1 = visualize_model_output(
                        rochet_net,
                        resolution=args.resolution,
                        item=1,
                        c=(0.5, 0.5),
                        l=0.0,
                    )

                    fig_c_center_l_half_item_0 = visualize_model_output(
                        rochet_net,
                        resolution=args.resolution,
                        item=0,
                        c=(0.5, 0.5),
                        l=0.5,
                    )
                    fig_c_center_l_half_item_1 = visualize_model_output(
                        rochet_net,
                        resolution=args.resolution,
                        item=1,
                        c=(0.5, 0.5),
                        l=0.5,
                    )

                    fig_c_third_l_0_item_0 = visualize_model_output(
                        rochet_net,
                        resolution=args.resolution,
                        item=0,
                        c=(1 / 3, 1 / 3),
                        l=0.0,
                    )
                    fig_c_third_l_0_item_1 = visualize_model_output(
                        rochet_net,
                        resolution=args.resolution,
                        item=1,
                        c=(1 / 3, 1 / 3),
                        l=0.0,
                    )

            if args.num_items == 2:
                c_center_l_0_item_0_path = Path(scratch_plots) / Path(
                    f"c_center_l_0_item_0_eval_{n}.png"
                )
                c_center_l_0_item_1_path = Path(scratch_plots) / Path(
                    f"c_center_l_0_item_1_eval_{n}.png"
                )
                fig_c_center_l_0_item_0.savefig(c_center_l_0_item_0_path)
                fig_c_center_l_0_item_1.savefig(c_center_l_0_item_1_path)
                plt.close()
                plt.close()

                c_center_l_half_item_0_path = Path(scratch_plots) / Path(
                    f"c_center_l_half_item_0_eval_{n}.png"
                )
                c_center_l_half_item_1_path = Path(scratch_plots) / Path(
                    f"c_center_l_half_item_1_eval_{n}.png"
                )
                fig_c_center_l_half_item_0.savefig(c_center_l_half_item_0_path)
                fig_c_center_l_half_item_1.savefig(c_center_l_half_item_1_path)
                plt.close()
                plt.close()

                c_third_l_0_item_0_path = Path(scratch_plots) / Path(
                    f"c_third_l_0_item_0_eval_{n}.png"
                )
                c_third_l_0_item_1_path = Path(scratch_plots) / Path(
                    f"c_third_l_0_item_1_eval_{n}.png"
                )
                fig_c_third_l_0_item_0.savefig(c_third_l_0_item_0_path)
                fig_c_third_l_0_item_1.savefig(c_third_l_0_item_1_path)
                plt.close()
                plt.close()

                wandb.log(
                    {
                        "eval_loss": eval_loss.item(),
                        "model_state": rochet_net.state_dict(),
                        "fig_c_center_l_0_item_0": wandb.Image(
                            str(c_center_l_0_item_0_path)
                        ),
                        "fig_c_center_l_0_item_1": wandb.Image(
                            str(c_center_l_0_item_1_path)
                        ),
                        "fig_c_center_l_half_item_0": wandb.Image(
                            str(c_center_l_half_item_0_path)
                        ),
                        "fig_c_center_l_half_item_1": wandb.Image(
                            str(c_center_l_half_item_1_path)
                        ),
                        "fig_c_third_l_0_item_0": wandb.Image(
                            str(c_third_l_0_item_0_path)
                        ),
                        "fig_c_third_l_0_item_1": wandb.Image(
                            str(c_third_l_0_item_1_path)
                        ),
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
        fig_c_center_l_0_item_0 = visualize_model_output(
            rochet_net,
            resolution=args.resolution,
            item=0,
            c=(0.5, 0.5),
            l=0.0,
        )
        fig_c_center_l_0_item_1 = visualize_model_output(
            rochet_net,
            resolution=args.resolution,
            item=1,
            c=(0.5, 0.5),
            l=0.0,
        )

        fig_c_center_l_half_item_0 = visualize_model_output(
            rochet_net,
            resolution=args.resolution,
            item=0,
            c=(0.5, 0.5),
            l=0.5,
        )
        fig_c_center_l_half_item_1 = visualize_model_output(
            rochet_net,
            resolution=args.resolution,
            item=1,
            c=(0.5, 0.5),
            l=0.5,
        )

        fig_c_third_l_0_item_0 = visualize_model_output(
            rochet_net,
            resolution=args.resolution,
            item=0,
            c=(1 / 3, 1 / 3),
            l=0.0,
        )
        fig_c_third_l_0_item_1 = visualize_model_output(
            rochet_net,
            resolution=args.resolution,
            item=1,
            c=(1 / 3, 1 / 3),
            l=0.0,
        )

        # Save the images and model parameters locally and to wandb

        c_center_l_0_item_0_path_final = Path(scratch_plots) / Path(
            f"c_center_l_0_item_0_final.png"
        )
        c_center_l_0_item_1_path_final = Path(scratch_plots) / Path(
            f"c_center_l_0_item_1_final.png"
        )

        c_center_l_half_item_0_path_final = Path(scratch_plots) / Path(
            f"c_center_l_half_item_0_final.png"
        )
        c_center_l_half_item_1_path_final = Path(scratch_plots) / Path(
            f"c_center_l_half_item_1_final.png"
        )

        c_third_l_0_item_0_path_final = Path(scratch_plots) / Path(
            f"c_third_l_0_item_0_final.png"
        )
        c_third_l_0_item_1_path_final = Path(scratch_plots) / Path(
            f"c_third_l_0_item_1_final.png"
        )

        fig_c_center_l_0_item_0.savefig(c_center_l_0_item_0_path_final)
        fig_c_center_l_0_item_1.savefig(c_center_l_0_item_1_path_final)

        fig_c_center_l_half_item_0.savefig(c_center_l_half_item_0_path_final)
        fig_c_center_l_half_item_1.savefig(c_center_l_half_item_1_path_final)

        fig_c_third_l_0_item_0.savefig(c_third_l_0_item_0_path_final)
        fig_c_third_l_0_item_1.savefig(c_third_l_0_item_1_path_final)

        wandb.log(
            {
                "fig_c_center_l_0_item_0_final": wandb.Image(
                    str(c_center_l_0_item_0_path_final)
                ),
                "fig_c_center_l_0_item_1_final": wandb.Image(
                    str(c_center_l_0_item_1_path_final)
                ),
                "fig_c_center_l_half_item_0_final": wandb.Image(
                    str(c_center_l_half_item_0_path_final)
                ),
                "fig_c_center_l_half_item_1_final": wandb.Image(
                    str(c_center_l_half_item_1_path_final)
                ),
                "fig_c_third_l_0_item_0_final": wandb.Image(
                    str(c_third_l_0_item_0_path_final)
                ),
                "fig_c_third_l_0_item_1_final": wandb.Image(
                    str(c_third_l_0_item_1_path_final)
                ),
                "final_model_state": rochet_net.state_dict(),
            }
        )
    else:
        wandb.log({"final_model_state": rochet_net.state_dict()})

    model_weights_path = Path("model_weights.pth")
    torch.save(rochet_net.cpu().state_dict(), model_weights_path)
    torch.save(rochet_net.cpu().state_dict(), Path(scratch_plots) / model_weights_path)
    wandb.save(str(model_weights_path))

    # Finish the run
    wandb.finish()
