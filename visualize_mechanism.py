import argparse
import wandb
from rochet import RochetNet
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import plotly.graph_objects as go


class FixedMenu:
    def __init__(self, allocs, payments):
        self.allocs = allocs
        self.payments = payments

    def __call__(self, x, train=False):
        expected_utility = (
            torch.sum(self.allocs[None, ...] * x[..., None, :], dim=-1)
            - self.payments[None, ...]
        )
        chosen_ind = torch.argmax(expected_utility, dim=-1)
        chosen_alloc = torch.index_select(self.allocs, 0, chosen_ind.flatten()).reshape(
            *chosen_ind.shape, -1
        )
        chosen_payment = torch.index_select(
            self.payments, 0, chosen_ind.flatten()
        ).reshape(*chosen_ind.shape)

        return chosen_alloc, chosen_payment


def download_model(run_path):
    file_name = "model_weights.pth"
    api = wandb.Api()
    run = api.run(run_path)
    print("Download started")
    run.file(file_name).download(replace=True)
    print("Download completed")

    state_dict = torch.load(file_name)
    num_menus = state_dict["alloc_param"].shape[0]
    num_items = state_dict["alloc_param"].shape[1] - 1
    model = RochetNet(num_items, num_menus)
    model.load_state_dict(state_dict)
    return model


def make_heatmap(model, ax, resolution=1000, is_payment=False, item=0, cmap="bwr"):
    x_range = np.linspace(0, 1, resolution)
    y_range = np.linspace(0, 1, resolution)

    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.stack([xx, yy], axis=-1)
    with torch.no_grad():
        grid_points = torch.tensor(grid_points, dtype=torch.float32, device="cpu")
        allocs, payments = model(grid_points, train=False)
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

    plt.colorbar(im, ax=ax, shrink=0.5, ticks=ticks)
    return im


def make_heatmap_3d(model, ax, resolution=50, is_payment=False, item=0, cmap="bwr"):
    x_range = np.linspace(0, 1, resolution)
    y_range = np.linspace(0, 1, resolution)
    z_range = np.linspace(0, 1, resolution)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    grid_points = np.stack([xx, yy, zz], axis=-1)
    with torch.no_grad():
        grid_points = torch.tensor(grid_points, dtype=torch.float32, device="cpu")
        allocs, payments = model(grid_points, train=False)
    if is_payment:
        heatmap = payments
    else:
        heatmap = allocs[..., item]
    if is_payment:
        vmin, vmax = -2, 2
    else:
        vmin, vmax = -1, 1
    heatmap = heatmap.cpu().numpy()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap)
    alphas = np.where(np.abs(heatmap) == 0.0, 0, 0.8)
    rgb_colors = cmap(norm(heatmap))[..., :3]
    rgba_colors = np.concatenate([rgb_colors, alphas[..., None]], axis=-1)

    im = ax.scatter(
        xx.flatten(),
        yy.flatten(),
        zz.flatten(),
        c=rgba_colors.reshape((-1, 4)),
        s=1,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ticks = [vmin, 0, vmax]

    plt.colorbar(im, ax=ax, shrink=0.5, ticks=ticks)
    return im


def make_heatmap_3d_plotly(
    model, resolution=50, is_payment=False, item=0, cmap="RdBu_r"
):
    x_range = np.linspace(0, 1, resolution)
    y_range = np.linspace(0, 1, resolution)
    z_range = np.linspace(0, 1, resolution)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    grid_points = np.stack([xx, yy, zz], axis=-1)
    with torch.no_grad():
        grid_points = torch.tensor(grid_points, dtype=torch.float32, device="cpu")
        allocs, payments = model(grid_points, train=False)
    if is_payment:
        heatmap = payments
        title = "Payment"
    else:
        heatmap = allocs[..., item]
        title = f"Item {item + 1} Allocation"
    if is_payment:
        vmin, vmax = -2, 2
    else:
        vmin, vmax = -1, 1
    heatmap = heatmap.cpu().numpy()
    fig = go.Figure(
        data=go.Volume(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=heatmap.flatten(),
            isomin=vmin,  # Minimum value to display
            isomax=vmax,  # Maximum value to display
            opacity=0.2,  # Opacity of the surfaces
            surface_count=10,  # Number of isosurfaces
            colorscale=cmap,  # Color scale
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Item 1 Value",
            yaxis_title="Item 2 Value",
            zaxis_title="Item 3 Value",
        ),
        font=dict(
            size=15,  # Set the font size here
        ),
        title_text=title,
        title_x=0.5,
        title_y=0.8,
    )
    fig.data[0].colorbar.x = 0.67
    fig.data[0].colorbar.len = 0.4
    fig.show()
    # fig.write_image("{}.pdf".format(title))


def process_run_path(run_path, extra_info=None):
    print(f"Processing run_path: {run_path}")
    model = download_model(run_path)

    if model.num_items == 2:
        font = {"size": 13}
        matplotlib.rc("font", **font)
        cmap = "bwr"
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        ax[0].set_title("Item 1 Allocation")
        make_heatmap(model, ax[0], is_payment=False, item=0, cmap=cmap)
        ax[1].set_title("Item 2 Allocation")
        make_heatmap(model, ax[1], is_payment=False, item=1, cmap=cmap)
        ax[2].set_title("Payment")
        make_heatmap(model, ax[2], is_payment=True, cmap=cmap)
        plt.savefig(f"{run_path.replace('/', '_')}_{extra_info}.png", dpi=500)
        plt.close()
    elif model.num_items == 3:
        font = {"size": 10}
        matplotlib.rc("font", **font)
        cmap = "bwr"
        cmap_plotly = "RdBu_r"
        fig, ax = plt.subplots(
            1,
            4,
            figsize=(20, 5),
            constrained_layout=True,
            subplot_kw={"projection": "3d"},
        )
        ax[0].set_title("Item 1 Allocation")
        make_heatmap_3d(model, ax[0], is_payment=False, item=0, cmap=cmap)
        make_heatmap_3d_plotly(model, is_payment=False, item=0, cmap=cmap_plotly)
        ax[1].set_title("Item 2 Allocation")
        make_heatmap_3d(model, ax[1], is_payment=False, item=1, cmap=cmap)
        make_heatmap_3d_plotly(model, is_payment=False, item=1, cmap=cmap_plotly)
        ax[2].set_title("Item 3 Allocation")
        make_heatmap_3d(model, ax[2], is_payment=False, item=2, cmap=cmap)
        make_heatmap_3d_plotly(model, is_payment=False, item=2, cmap=cmap_plotly)
        ax[3].set_title("Payment")
        make_heatmap_3d(model, ax[3], is_payment=True, cmap=cmap)
        make_heatmap_3d_plotly(model, is_payment=True, cmap=cmap_plotly)
        # plt.show()
        plt.savefig(f"{run_path.replace('/', '_')}_{extra_info}.png", dpi=500)
        plt.close()
    else:
        raise ValueError("Model must have 2 or 3 items.")


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


def get_fixed_menu_model_c_center(l):
    model = FixedMenu(
        torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, -1],
                [1, -1],
                [-1, 1],
                [0, 0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                (3 * l + 2) / (4 * l + 2),
                (3 * l + 2) / (4 * l + 2),
                -l / (4 * l + 2),
                -l / (4 * l + 2),
                (-(2**0.5) * l + 6 * l + 4) / (4 * l + 2),
                -(2 + 2**0.5) * l / (4 * l + 2),
                (-(2**0.5) * l + 2 * l + 2) / (4 * l + 2),
                (-(2**0.5) * l + 2 * l + 2) / (4 * l + 2),
                0,
            ],
            dtype=torch.float32,
        ),
    )
    return model


def get_fixed_menu_model_c_off_center():
    model = FixedMenu(
        torch.tensor(
            [
                [-1, 0],
                [0, -1],
                [-1, -1],
                [1, -1],
                [-1, 1],
                [1, 1],
                [0, 0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                -1 / 9,
                -1 / 9,
                (-2 - 2**0.5) / 9,
                2
                / 63
                * (
                    -((2 * (67 - 41 * 2**0.5)) ** 0.5)
                    - 9 * 2**0.5
                    + (2 * (155 + 107 * 2**0.5)) ** 0.5
                    + 6
                ),
                2
                / 63
                * (
                    -((2 * (67 - 41 * 2**0.5)) ** 0.5)
                    - 9 * 2**0.5
                    + (2 * (155 + 107 * 2**0.5)) ** 0.5
                    + 6
                ),
                2
                / 63
                * (
                    (2 * (67 - 41 * 2**0.5)) ** 0.5
                    - 15 * 2**0.5
                    + (2 * (155 + 107 * 2**0.5)) ** 0.5
                    + 31
                ),
                0,
            ],
            dtype=torch.float32,
        ),
    )
    return model


def plot_fixed_menu_2d(model, extra_info):
    font = {"size": 13}
    matplotlib.rc("font", **font)
    cmap = "bwr"
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2, wspace=0.2)
    ax[0].set_title("Item 1 Allocation")
    make_heatmap(model, ax[0], is_payment=False, item=0, cmap=cmap)
    ax[1].set_title("Item 2 Allocation")
    make_heatmap(model, ax[1], is_payment=False, item=1, cmap=cmap)
    ax[2].set_title("Payment")
    make_heatmap(model, ax[2], is_payment=True, cmap=cmap)
    plt.savefig("fixed_menu_{}.png".format(extra_info), dpi=500)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download model and generate heatmaps."
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
