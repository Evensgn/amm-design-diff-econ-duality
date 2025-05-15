import argparse
import wandb
from rochet_parameterized import RochetNet
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import plotly.graph_objects as go
from pathlib import Path
import os


def get_layer_sizes_of_sequential(state_dict):
    """Get the layer sizes of an nn.Sequential model from a state dict."""
    layer_sizes_dict = {}

    for key, weight in state_dict.items():
        if "weight" in key:
            input_size = weight.shape[1]
            output_size = weight.shape[0]
            # Extract the layer number from the key
            layer_num = int(key.split(".")[1])
            layer_sizes_dict[layer_num] = (input_size, output_size)

    # Sort the layer sizes by layer number and return as a size list
    layer_sizes = []
    for layer_num in sorted(layer_sizes_dict.keys()):
        layer_sizes.append(layer_sizes_dict[layer_num][0])
        if layer_num == max(layer_sizes_dict.keys()):
            layer_sizes.append(layer_sizes_dict[layer_num][1])

    return layer_sizes


def download_model(run_path):
    file_name = "model_weights.pth"
    api = wandb.Api()
    run = api.run(run_path)
    print("Download started")
    run.file(file_name).download(replace=True)
    print("Download completed")

    state_dict = torch.load(file_name)
    layer_sizes = get_layer_sizes_of_sequential(state_dict)
    num_items = layer_sizes[0] - 1
    num_menus = layer_sizes[-1] // (num_items + 2)
    hidden_layer_sizes = layer_sizes[1:-1]

    print(f"num_items: {num_items}, num_menus: {num_menus}")
    model = RochetNet(num_items, num_menus, hidden_layer_sizes)
    model.load_state_dict(state_dict)
    return model


def make_heatmap(
    model,
    ax,
    resolution=1000,
    is_payment=False,
    item=0,
    cmap="bwr",
    c=(0.5, 0.5),
    l=0.0,
    highlight_c=False,
    highlight_x=None,
):
    x_range = np.linspace(0, 1, resolution)
    y_range = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    device = model.menu_model.parameters().__next__().device

    num_grid_points = grid_points.shape[0]
    batch_size = 200000

    allocs_list = []
    payments_list = []

    with torch.no_grad():
        for i in range(0, num_grid_points, batch_size):
            batch_grid_points = grid_points[i : i + batch_size]
            batch_c_tensor = torch.tensor(
                [c] * batch_grid_points.shape[0],
                dtype=torch.float32,
                device=device,
            )
            batch_l_tensor = (
                torch.ones(batch_grid_points.shape[0], 1, device=device) * l
            )
            batch_grid_points = torch.tensor(
                batch_grid_points, dtype=torch.float32, device=device
            )
            batch_allocs, batch_payments = model(
                batch_grid_points, batch_c_tensor, batch_l_tensor, train=False
            )

            allocs_list.append(batch_allocs.cpu())
            payments_list.append(batch_payments.cpu())

    allocs = torch.cat(allocs_list, dim=0).reshape(resolution, resolution, -1)
    payments = torch.cat(payments_list, dim=0).reshape(resolution, resolution)

    if is_payment:
        heatmap = payments
    else:
        heatmap = allocs[..., item]
    if is_payment:
        vmin, vmax = -2, 2
    else:
        vmin, vmax = -1, 1
    heatmap = heatmap.cpu().numpy()
    im = ax.pcolormesh(x_range, y_range, heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ticks = [vmin, 0, vmax]

    if highlight_c:
        # Highlight the c point with an orange dot
        ax.plot(c[0], c[1], "o", color="orange", markersize=6)

    if highlight_x is not None:
        # Highlight the x point with a black cross
        ax.plot(highlight_x[0], highlight_x[1], "x", color="black", markersize=10)

    plt.colorbar(im, ax=ax, shrink=0.5, ticks=ticks)
    return im


def evaluate_profit(model, num_samples, dist_generator, c=(0.5, 0.5), l=0.0):
    bidder_values = dist_generator(num_samples)

    device = model.menu_model.parameters().__next__().device

    batch_size = 200000

    allocs_list = []
    payments_list = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_bidder_values = bidder_values[i : i + batch_size]
            batch_c_tensor = torch.tensor(
                [c] * batch_bidder_values.shape[0],
                dtype=torch.float32,
                device=device,
            )
            batch_l_tensor = (
                torch.ones(batch_bidder_values.shape[0], 1, device=device) * l
            )
            batch_bidder_values = torch.tensor(
                batch_bidder_values, dtype=torch.float32, device=device
            )
            batch_allocs, batch_payments = model(
                batch_bidder_values, batch_c_tensor, batch_l_tensor, train=False
            )

            allocs_list.append(batch_allocs.cpu())
            payments_list.append(batch_payments.cpu())

    allocs = torch.cat(allocs_list, dim=0)
    payments = torch.cat(payments_list, dim=0)
    profit = payments - torch.einsum(
        "bn,bn->b",
        allocs,
        torch.tensor([c] * allocs.shape[0], dtype=torch.float32, device=device),
    )
    profit_mean = profit.mean().item()
    profit_std = profit.std().item()
    print("profit_mean:", profit_mean)
    print("profit_std:", profit_std)
    print("num_samples:", profit.shape[0])
    return profit_mean, profit_std


def process_model(
    model,
    run_path,
    extra_info=None,
    c=(0.5, 0.5),
    l=0.0,
    highlight_c=False,
    highlight_x=None,
):
    font = {"size": 13}
    matplotlib.rc("font", **font)
    cmap = "bwr"
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax[0].set_title("Item 1 Allocation")
    make_heatmap(
        model,
        ax[0],
        is_payment=False,
        item=0,
        cmap=cmap,
        c=c,
        l=l,
        highlight_c=highlight_c,
        highlight_x=highlight_x,
    )
    ax[1].set_title("Item 2 Allocation")
    make_heatmap(
        model,
        ax[1],
        is_payment=False,
        item=1,
        cmap=cmap,
        c=c,
        l=l,
        highlight_c=highlight_c,
        highlight_x=highlight_x,
    )
    ax[2].set_title("Payment")
    make_heatmap(
        model,
        ax[2],
        is_payment=True,
        cmap=cmap,
        c=c,
        l=l,
        highlight_c=highlight_c,
        highlight_x=highlight_x,
    )
    plt.savefig(
        f"parameterized_plots/{run_path.replace('/', '_')}_c_{c}_l_{l}_{extra_info}.png",
        dpi=500,
    )
    plt.close()


def process_run_path(run_path, extra_info=None):
    print(f"Processing run_path: {run_path}")
    os.makedirs("parameterized_plots", exist_ok=True)

    model = download_model(run_path)

    if model.num_items == 2:
        # Examples of how to evaluate and visualize a parameterized network

        # Example: numerically evaluate the profit of a parameterized network for a specific combination of c and l
        num_samples_for_profit = 10**7
        uniform_dist_generator = lambda size_arg: torch.rand(
            size_arg,
            model.num_items,
            device=model.menu_model.parameters().__next__().device,
        )
        profit_mean, profit_std = evaluate_profit(
            model, num_samples_for_profit, uniform_dist_generator, c=(1 / 3, 1 / 3), l=0
        )
        # Save the profit to a file
        with open("parameterized_plots/profits.txt", "a") as file:
            file.write(
                f"{run_path}, c=(1/3, 1/3), l=0, profit_mean={profit_mean}, profit_std={profit_std}, num_samples={num_samples_for_profit}\n"
            )

        # Example: fix c=(0.5, 0.5) and vary the value of l
        for l_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            process_model(
                model, run_path, extra_info="fix_c_vary_l", c=(0.5, 0.5), l=l_value
            )
    else:
        raise ValueError("Model must have 2 items.")


def process_file(file_path):
    # Try to auto-detect the file format by checking the first line for a comma
    with open(file_path, "r") as file:
        first_line = file.readline()
        file.seek(0)  # Reset file read position to the beginning

        if "," in first_line:
            # Process as CSV
            csvreader = csv.reader(file)
            for row in csvreader:
                run_path, extra_info = row[0], row[1]
                process_run_path(run_path, extra_info)
        else:
            # Process as plain text
            run_paths = file.read().splitlines()
            for run_path in run_paths:
                process_run_path(run_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download model and generate heatmaps for a parameterized network."
    )
    parser.add_argument("--run_path", type=str, help="Single run path for the model.")
    parser.add_argument(
        "--many_run_paths",
        type=str,
        help="File containing multiple run paths, one per line.",
    )

    args = parser.parse_args()

    if args.run_path:
        process_run_path(args.run_path)
    elif args.many_run_paths:
        process_file(args.many_run_paths)
    else:
        print("No run_path or many_run_paths argument provided.")


if __name__ == "__main__":
    main()
