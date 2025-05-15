# Portions of this code, such as the functions for calculating the
# source and target distributions for settings where trader values are sampled
# from a beta distribution, were initially generated with the help of an LLM.

import numpy as np
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad


# Four directions: right, up, left, down.
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
RIGHT, UP, LEFT, DOWN = range(4)
# Two pairs of opposite directions: (right, left) and (up, down).
LEFT_RIGHT, UP_DOWN = range(2)
LR_UP_DOWN, LR_DOWN_UP = range(2)
UD_LEFT_RIGHT, UD_RIGHT_LEFT = range(2)


def optimal_transport_lp(lambda_, c, N=1001, trader_value_dist="uniform"):
    # Discretize the [0, 1]^2 square into N^2 grid points.
    grid_1d = np.linspace(0, 1, N)
    grid_points = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1, 2)

    # Index of the point mass at c.
    idx_c = np.argmin(np.sum((grid_points - c) ** 2, axis=1))

    # Index of the points on the right boundary.
    idx_right_boundary = np.where(grid_points[:, 0] == 1)[0]
    # Index of the points on the top boundary.
    idx_top_boundary = np.where(grid_points[:, 1] == 1)[0]
    # Index of the points on the left boundary.
    idx_left_boundary = np.where(grid_points[:, 0] == 0)[0]
    # Index of the points on the bottom boundary.
    idx_bottom_boundary = np.where(grid_points[:, 1] == 0)[0]
    # Index of the points in the interior.
    idx_interior = np.where(
        (grid_points[:, 0] != 0)
        & (grid_points[:, 0] != 1)
        & (grid_points[:, 1] != 0)
        & (grid_points[:, 1] != 1)
    )[0]

    # Define the source distribution and the target distribution.
    if trader_value_dist == "uniform":
        # Source distribution:
        #     a point mass of +1 at c
        #     mass of +(1 - c_1) * lambda_ on the right boundary
        #     mass of +(1 - c_2) * lambda_ on the top boundary
        #     mass of +c_1 * lambda_ on the left boundary
        #     mass of +c_2 * lambda_ on the bottom boundary
        source_distribution = np.zeros(N**2)
        source_distribution[idx_c] += 1.0
        source_distribution[idx_right_boundary] += (
            (1 - c[0]) * lambda_ / len(idx_right_boundary)
        )
        source_distribution[idx_top_boundary] += (
            (1 - c[1]) * lambda_ / len(idx_top_boundary)
        )
        source_distribution[idx_left_boundary] += (
            c[0] * lambda_ / len(idx_left_boundary)
        )
        source_distribution[idx_bottom_boundary] += (
            c[1] * lambda_ / len(idx_bottom_boundary)
        )

        # Target distribution: +(2 * lambda_ + 1) of mass uniformly distributed on the
        # grid points.
        target_distribution = (2 * lambda_ + 1) * np.ones(N**2) / (N**2)
    elif trader_value_dist == "beta12" or trader_value_dist == "beta22":
        if lambda_ != 1:
            raise ValueError(
                f"Unsupported lambda_ value for beta distribution: {lambda_}"
            )
        (
            source_distribution,
            target_distribution,
        ) = compute_discrete_distributions_for_beta(
            N - 1, c[0], c[1], trader_value_dist
        )
    else:
        raise ValueError(f"Unsupported trader_value_dist value: {trader_value_dist}")

    # Define the LP variables of local flow in 4 directions from each grid point.
    model = Model("optimal_transport")
    flow = model.addVars(N**2, 4, lb=0.0, vtype=GRB.CONTINUOUS, name="flow")
    # Some flow directions are not allowed for the boundary points.
    for idx in idx_right_boundary:
        # flow[idx, 0] has to be zero.
        model.addConstr(flow[idx, 0] == 0.0, name=f"right_boundary_{idx}")
    for idx in idx_top_boundary:
        # flow[idx, 1] has to be zero.
        model.addConstr(flow[idx, 1] == 0.0, name=f"top_boundary_{idx}")
    for idx in idx_left_boundary:
        # flow[idx, 2] has to be zero.
        model.addConstr(flow[idx, 2] == 0.0, name=f"left_boundary_{idx}")
    for idx in idx_bottom_boundary:
        # flow[idx, 3] has to be zero.
        model.addConstr(flow[idx, 3] == 0.0, name=f"bottom_boundary_{idx}")

    # The cost of moving one unit of mass in each direction from each grid point is by
    # default the Manhattan distance between the two grid points.
    cost = np.ones((N**2, 4)) / (N - 1)
    # Moving away from c is free.
    for idx in range(N**2):
        for d in range(4):
            # Check if the direction is moving away from c.
            if np.dot(DIRECTIONS[d], grid_points[idx] - c) >= 0:
                cost[idx, d] = 0.0

    # A mean-preserving spread is also free.
    # Supporting general mean-preserving spread might require O(N^4) variables, so we do
    # only support local mean-preserving spread for now.
    # For each grid point, add two variables for the mean-preserving spread in the two
    # pairs of opposite directions: (right, left) and (up, down).
    local_balayage = model.addVars(
        N**2, 2, lb=0.0, vtype=GRB.CONTINUOUS, name="balayage"
    )
    # Some balayage directions are not allowed for the boundary points.
    for idx in range(N**2):
        if idx in idx_right_boundary:
            # local_balayage[idx, LEFT_RIGHT] has to be zero.
            model.addConstr(
                local_balayage[idx, LEFT_RIGHT] == 0.0, name=f"right_balayage_{idx}"
            )
        if idx in idx_top_boundary:
            # local_balayage[idx, UP_DOWN] has to be zero.
            model.addConstr(
                local_balayage[idx, UP_DOWN] == 0.0, name=f"top_balayage_{idx}"
            )
        if idx in idx_left_boundary:
            # local_balayage[idx, LEFT_RIGHT] has to be zero.
            model.addConstr(
                local_balayage[idx, LEFT_RIGHT] == 0.0, name=f"left_balayage_{idx}"
            )
        if idx in idx_bottom_boundary:
            # local_balayage[idx, UP_DOWN] has to be zero.
            model.addConstr(
                local_balayage[idx, UP_DOWN] == 0.0, name=f"bottom_balayage_{idx}"
            )
        if idx in idx_interior:
            # No balayage for the interior points.
            model.addConstr(
                local_balayage[idx, LEFT_RIGHT] == 0.0,
                name=f"interior_balayage_{idx}",
            )
            model.addConstr(
                local_balayage[idx, UP_DOWN] == 0.0,
                name=f"interior_balayage_{idx}",
            )

    # Mean-preserving spread between a pair of grid points on opposite boundaries.
    left_right_boundary_balayage = model.addVars(
        N**2, 2, lb=0.0, vtype=GRB.CONTINUOUS, name="left_right_boundary_balayage"
    )
    up_down_boundary_balayage = model.addVars(
        N**2, 2, lb=0.0, vtype=GRB.CONTINUOUS, name="up_down_boundary_balayage"
    )
    # Only the spreading directions are valid balayage directions.
    for idx_l in idx_left_boundary:
        for idx_r in idx_right_boundary:
            if (idx_l % N) < (idx_r % N) or (idx_l % N) == N - 1 or (idx_r % N) == 0:
                model.addConstr(
                    left_right_boundary_balayage[
                        (idx_l % N) * N + (idx_r % N), LR_UP_DOWN
                    ]
                    == 0.0,
                    name=f"lr_up_down_boundary_balayage_{idx_l}_{idx_r}",
                )
            if (idx_l % N) > (idx_r % N) or (idx_l % N) == 0 or (idx_r % N) == N - 1:
                model.addConstr(
                    left_right_boundary_balayage[
                        (idx_l % N) * N + (idx_r % N), LR_DOWN_UP
                    ]
                    == 0.0,
                    name=f"lr_down_up_boundary_balayage_{idx_l}_{idx_r}",
                )

    for idx_u in idx_top_boundary:
        for idx_d in idx_bottom_boundary:
            if (
                (idx_u // N) < (idx_d // N)
                or (idx_u // N) == N - 1
                or (idx_d // N) == 0
            ):
                model.addConstr(
                    up_down_boundary_balayage[
                        (idx_u // N) * N + (idx_d // N), UD_RIGHT_LEFT
                    ]
                    == 0.0,
                    name=f"ud_right_left_boundary_balayage_{idx_u}_{idx_d}",
                )
            if (
                (idx_u // N) > (idx_d // N)
                or (idx_u // N) == 0
                or (idx_d // N) == N - 1
            ):
                model.addConstr(
                    up_down_boundary_balayage[
                        (idx_u // N) * N + (idx_d // N), UD_LEFT_RIGHT
                    ]
                    == 0.0,
                    name=f"ud_left_right_boundary_balayage_{idx_u}_{idx_d}",
                )

    # Total flow and total preprocessed flow should satisfy the source and target
    # distributions.
    for idx in range(N**2):
        # Check for bottom-left corner.
        if idx == 0:
            neighbors = [(idx + 1, DOWN), (idx + N, LEFT)]
            local_balayage_neighbors = [(idx + 1, UP_DOWN), (idx + N, LEFT_RIGHT)]

            left_right_boundary_balayage_neighbors = []
            for idx_r in idx_right_boundary:
                left_right_boundary_balayage_neighbors.append(
                    (((idx + 1) % N) * N + (idx_r % N), LR_DOWN_UP)
                )
            up_down_boundary_balayage_neighbors = []
            for idx_u in idx_top_boundary:
                up_down_boundary_balayage_neighbors.append(
                    ((idx_u // N) * N + ((idx + N) // N), UD_RIGHT_LEFT)
                )
        # Check for bottom-right corner.
        elif idx == N * (N - 1):
            neighbors = [(idx - N, RIGHT), (idx + 1, DOWN)]
            local_balayage_neighbors = [(idx - N, LEFT_RIGHT), (idx + 1, UP_DOWN)]

            left_right_boundary_balayage_neighbors = []
            for idx_l in idx_left_boundary:
                left_right_boundary_balayage_neighbors.append(
                    ((idx_l % N) * N + ((idx + 1) % N), LR_UP_DOWN)
                )
            up_down_boundary_balayage_neighbors = []
            for idx_u in idx_top_boundary:
                up_down_boundary_balayage_neighbors.append(
                    ((idx_u // N) * N + ((idx - N) // N), UD_LEFT_RIGHT)
                )
        # Check for top-left corner.
        elif idx == N - 1:
            neighbors = [(idx - 1, UP), (idx + N, LEFT)]
            local_balayage_neighbors = [(idx - 1, UP_DOWN), (idx + N, LEFT_RIGHT)]

            left_right_boundary_balayage_neighbors = []
            for idx_r in idx_right_boundary:
                left_right_boundary_balayage_neighbors.append(
                    (((idx - 1) % N) * N + (idx_r % N), LR_UP_DOWN)
                )
            up_down_boundary_balayage_neighbors = []
            for idx_d in idx_bottom_boundary:
                up_down_boundary_balayage_neighbors.append(
                    (((idx + N) // N) * N + (idx_d // N), UD_LEFT_RIGHT)
                )
        # Check for top-right corner.
        elif idx == N**2 - 1:
            neighbors = [(idx - 1, UP), (idx - N, RIGHT)]
            local_balayage_neighbors = [(idx - 1, UP_DOWN), (idx - N, LEFT_RIGHT)]

            left_right_boundary_balayage_neighbors = []
            for idx_l in idx_left_boundary:
                left_right_boundary_balayage_neighbors.append(
                    ((idx_l % N) * N + ((idx - 1) % N), LR_DOWN_UP)
                )
            up_down_boundary_balayage_neighbors = []
            for idx_d in idx_bottom_boundary:
                up_down_boundary_balayage_neighbors.append(
                    (((idx - N) // N) * N + (idx_d // N), UD_RIGHT_LEFT)
                )
        # Check for bottom boundary.
        elif idx % N == 0:
            neighbors = [(idx + 1, DOWN), (idx + N, LEFT), (idx - N, RIGHT)]
            local_balayage_neighbors = [
                (idx + 1, UP_DOWN),
                (idx + N, LEFT_RIGHT),
                (idx - N, LEFT_RIGHT),
            ]

            left_right_boundary_balayage_neighbors = []
            up_down_boundary_balayage_neighbors = []
            for idx_u in idx_top_boundary:
                up_down_boundary_balayage_neighbors.append(
                    ((idx_u // N) * N + ((idx - N) // N), UD_LEFT_RIGHT)
                )
                up_down_boundary_balayage_neighbors.append(
                    ((idx_u // N) * N + ((idx + N) // N), UD_RIGHT_LEFT)
                )
        # Check for left boundary.
        elif idx < N:
            neighbors = [(idx + 1, DOWN), (idx + N, LEFT), (idx - 1, UP)]
            local_balayage_neighbors = [
                (idx + 1, UP_DOWN),
                (idx + N, LEFT_RIGHT),
                (idx - 1, UP_DOWN),
            ]

            left_right_boundary_balayage_neighbors = []
            for idx_r in idx_right_boundary:
                left_right_boundary_balayage_neighbors.append(
                    (((idx - 1) % N) * N + (idx_r % N), LR_UP_DOWN)
                )
                left_right_boundary_balayage_neighbors.append(
                    (((idx + 1) % N) * N + (idx_r % N), LR_DOWN_UP)
                )
            up_down_boundary_balayage_neighbors = []
        # Check for top boundary.
        elif idx % N == N - 1:
            neighbors = [(idx - 1, UP), (idx + N, LEFT), (idx - N, RIGHT)]
            local_balayage_neighbors = [
                (idx - 1, UP_DOWN),
                (idx + N, LEFT_RIGHT),
                (idx - N, LEFT_RIGHT),
            ]

            left_right_boundary_balayage_neighbors = []
            up_down_boundary_balayage_neighbors = []
            for idx_d in idx_bottom_boundary:
                up_down_boundary_balayage_neighbors.append(
                    (((idx + N) // N) * N + (idx_d // N), UD_LEFT_RIGHT)
                )
                up_down_boundary_balayage_neighbors.append(
                    (((idx - N) // N) * N + (idx_d // N), UD_RIGHT_LEFT)
                )
        # Check for right boundary.
        elif idx > N * (N - 1):
            neighbors = [(idx - 1, UP), (idx + 1, DOWN), (idx - N, RIGHT)]
            local_balayage_neighbors = [
                (idx - 1, UP_DOWN),
                (idx + 1, UP_DOWN),
                (idx - N, LEFT_RIGHT),
            ]

            left_right_boundary_balayage_neighbors = []
            for idx_l in idx_left_boundary:
                left_right_boundary_balayage_neighbors.append(
                    ((idx_l % N) * N + ((idx - 1) % N), LR_DOWN_UP)
                )
                left_right_boundary_balayage_neighbors.append(
                    ((idx_l % N) * N + ((idx + 1) % N), LR_UP_DOWN)
                )
            up_down_boundary_balayage_neighbors = []
        else:
            neighbors = [
                (idx - 1, UP),
                (idx + 1, DOWN),
                (idx - N, RIGHT),
                (idx + N, LEFT),
            ]
            local_balayage_neighbors = [
                (idx - 1, UP_DOWN),
                (idx + 1, UP_DOWN),
                (idx - N, LEFT_RIGHT),
                (idx + N, LEFT_RIGHT),
            ]
            left_right_boundary_balayage_neighbors = []
            up_down_boundary_balayage_neighbors = []

        left_right_boundary_self_outflow_indices = []
        if idx in idx_left_boundary:
            for idx_r in idx_right_boundary:
                for d in range(2):
                    left_right_boundary_self_outflow_indices.append(
                        ((idx % N) * N + (idx_r % N), d)
                    )
        elif idx in idx_right_boundary:
            for idx_l in idx_left_boundary:
                for d in range(2):
                    left_right_boundary_self_outflow_indices.append(
                        ((idx_l % N) * N + (idx % N), d)
                    )

        up_down_boundary_self_outflow_indices = []
        if idx in idx_top_boundary:
            for idx_d in idx_bottom_boundary:
                for d in range(2):
                    up_down_boundary_self_outflow_indices.append(
                        ((idx // N) * N + (idx_d // N), d)
                    )
        elif idx in idx_bottom_boundary:
            for idx_u in idx_top_boundary:
                for d in range(2):
                    up_down_boundary_self_outflow_indices.append(
                        ((idx_u // N) * N + (idx // N), d)
                    )

        # Total flow away from each grid point and total flow into each grid point should
        # sum up to the difference between the source and target distributions.
        # print(f"flow_balance_{idx}")
        model.addConstr(
            sum(flow[n, d] for n, d in neighbors)
            + sum(local_balayage[n, d] for n, d in local_balayage_neighbors)
            + sum(
                left_right_boundary_balayage[n, d]
                for n, d in left_right_boundary_balayage_neighbors
            )
            + sum(
                up_down_boundary_balayage[n, d]
                for n, d in up_down_boundary_balayage_neighbors
            )
            - sum(flow[idx, d] for d in range(4))
            - 2 * sum(local_balayage[idx, d] for d in range(2))
            - sum(
                left_right_boundary_balayage[n, d]
                for n, d in left_right_boundary_self_outflow_indices
            )
            - sum(
                up_down_boundary_balayage[n, d]
                for n, d in up_down_boundary_self_outflow_indices
            )
            == target_distribution[idx] - source_distribution[idx],
            name=f"flow_balance_{idx}",
        )

    # Define the objective function.
    model.setObjective(
        sum(flow[idx, d] * cost[idx, d] for idx in range(N**2) for d in range(4)),
        GRB.MINIMIZE,
    )

    # Solve the LP.
    model.optimize()

    # Extract the optimal solution
    if model.status == GRB.OPTIMAL:
        flow_values = model.getAttr("x", flow)
        local_balayage_values = model.getAttr("x", local_balayage)
        left_right_boundary_balayage_values = model.getAttr(
            "x", left_right_boundary_balayage
        )
        up_down_boundary_balayage_values = model.getAttr("x", up_down_boundary_balayage)

        print(f"Total transportation cost: {model.objVal}")
        return (
            model.objVal,
            flow_values,
            (
                local_balayage_values,
                left_right_boundary_balayage_values,
                up_down_boundary_balayage_values,
            ),
        )
    else:
        print("No optimal solution found.")
        return None, None, None


def compute_cell_mass_beta12(i, j, N, c1, c2):
    """
    Compute the total mass of the continuous measure in cell C_{i,j} = [i/N, (i+1)/N] x [j/N, (j+1)/N] for lambda = 1.
    This mass comes from three sources:
      1. The interior (bulk) density.
      2. Boundary contributions (if the cell touches a domain boundary).
      3. The atom at c, if c lies in the cell.
    """
    # Define cell boundaries.
    x_low = i / N
    x_high = (i + 1) / N
    y_low = j / N
    y_high = (j + 1) / N

    # ---------------------------
    # 1. Interior density integration.
    # The interior density is defined as:
    #   rho_int(x1,x2) = 4 * [ (1-x2)*(x1-c1) + (1-x1)*(x2-c2) - 3*(1-x1)*(1-x2) ]
    # Integrate over (x1,x2) in [x_low, x_high] x [y_low, y_high].
    def rho_int(x, y):
        return 4 * ((1 - y) * (x - c1) + (1 - x) * (y - c2) - 3 * (1 - x) * (1 - y))

    # dblquad expects the integrand as a function of y (inner) and x (outer).
    def integrand(y, x):
        return rho_int(x, y)

    mass_int, err_int = dblquad(
        integrand, x_low, x_high, lambda x: y_low, lambda x: y_high
    )

    # ---------------------------
    # 2. Boundary contributions.
    mass_bound = 0.0
    # Left boundary (x=0): present if i == 0.
    if i == 0:
        # The left-edge density: rho_left(y) = 4 * c_1 * (1 - y)
        def rho_left(y):
            return 4 * c1 * (1 - y)

        mass_left, err_left = quad(rho_left, y_low, y_high)
        mass_bound += mass_left
    # Bottom boundary (y=0): present if j == 0.
    if j == 0:
        # Bottom-edge density: rho_bottom(x) = 4 * c2 * (1 - x)
        def rho_bottom(x):
            return 4 * c2 * (1 - x)

        mass_bottom, err_bottom = quad(rho_bottom, x_low, x_high)
        mass_bound += mass_bottom
    # Top boundary (y=1): present if the cell touches the top, i.e. j == N-1.
    if j == N - 1:
        # Top-edge density: rho_top(x) = 0
        def rho_top(x):
            return 0

        mass_top, err_top = quad(rho_top, x_low, x_high)
        mass_bound += mass_top
    # Right boundary (x=1): present if the cell touches the right, i.e. i == N-1.
    if i == N - 1:
        # Right-edge density: rho_right(y) = 0
        def rho_right(y):
            return 0

        mass_right, err_right = quad(rho_right, y_low, y_high)
        mass_bound += mass_right

    # ---------------------------
    # 3. Atom at c: if the point c = (c1, c2) lies in this cell, add mass 1.
    mass_atom = 0.0
    if (x_low <= c1 < x_high) and (y_low <= c2 < y_high):
        mass_atom = 1.0

    # Total mass in cell = interior + boundary + atom.
    total_mass = mass_int + mass_bound + mass_atom
    return total_mass


# This function was generated with the help of an LLM.
def compute_cell_mass_beta22(i, j, N, c1, c2):
    """
    Compute the total mass of the continuous measure in cell C_{i,j} = [i/N, (i+1)/N] x [j/N, (j+1)/N] for lambda = 1.
    This mass comes from three sources:
      1. The interior (bulk) density.
      2. Boundary contributions (if the cell touches a domain boundary).
      3. The atom at c, if c lies in the cell.
    """
    # Define cell boundaries.
    x_low = i / N
    x_high = (i + 1) / N
    y_low = j / N
    y_high = (j + 1) / N

    # ---------------------------
    # 1. Interior density integration.
    # The interior density is defined as:
    #   rho_int(x1,x2) =  -36*( x2 (1 - x2) (1 - 2 x1) (x1 - c1) + x1 (1 - x1) (1 - 2 x2) (x2 - c2) + 3 x1 (1 - x1) x2 (1 - x2))
    # Integrate over (x1,x2) in [x_low, x_high] x [y_low, y_high].
    def rho_int(x, y):
        return -36 * (
            y * (1 - y) * (1 - 2 * x) * (x - c1)
            + x * (1 - x) * (1 - 2 * y) * (y - c2)
            + 3 * x * (1 - x) * y * (1 - y)
        )

    # dblquad expects the integrand as a function of y (inner) and x (outer).
    def integrand(y, x):
        return rho_int(x, y)

    mass_int, err_int = dblquad(
        integrand, x_low, x_high, lambda x: y_low, lambda x: y_high
    )

    # ---------------------------
    # 2. Boundary contributions.
    mass_bound = 0.0
    # Left boundary (x=0): present if i == 0.
    if i == 0:
        # The left-edge density: rho_left(y) = 0
        def rho_left(y):
            return 0

        mass_left, err_left = quad(rho_left, y_low, y_high)
        mass_bound += mass_left
    # Bottom boundary (y=0): present if j == 0.
    if j == 0:
        # Bottom-edge density: rho_bottom(x) = 0
        def rho_bottom(x):
            return 0

        mass_bottom, err_bottom = quad(rho_bottom, x_low, x_high)
        mass_bound += mass_bottom
    # Top boundary (y=1): present if the cell touches the top, i.e. j == N-1.
    if j == N - 1:
        # Top-edge density: rho_top(x) = 0
        def rho_top(x):
            return 0

        mass_top, err_top = quad(rho_top, x_low, x_high)
        mass_bound += mass_top
    # Right boundary (x=1): present if the cell touches the right, i.e. i == N-1.
    if i == N - 1:
        # Right-edge density: rho_right(y) = 0
        def rho_right(y):
            return 0

        mass_right, err_right = quad(rho_right, y_low, y_high)
        mass_bound += mass_right

    # ---------------------------
    # 3. Atom at c: if the point c = (c1, c2) lies in this cell, add mass 1.
    mass_atom = 0.0
    if (x_low <= c1 < x_high) and (y_low <= c2 < y_high):
        mass_atom = 1.0

    # Total mass in cell = interior + boundary + atom.
    total_mass = mass_int + mass_bound + mass_atom
    return total_mass


def compute_discrete_distributions_for_beta(N, c1, c2, trader_value_dist):
    """
    Discretize the continuous measure on [0,1]^2.
    Partition [0,1]^2 into N x N cells. The grid points (the bottom-left corners)
    are indexed by (i, j) for 0 <= i,j <= N-1. For grid points with i = N or j = N
    (i.e. those not serving as a bottom-left corner of any cell), assign mass zero.

    The function returns two flattened arrays (of length (N+1)^2) corresponding to the grid
    points in the same order as generated by:
       grid_1d = np.linspace(0,1,N+1)
       grid_points = np.array(np.meshgrid(grid_1d, grid_1d)).T.reshape(-1,2)
    The two arrays are:
       S: source distribution (positive mass)
       T: target distribution (absolute value of negative mass)
    """
    if trader_value_dist == "beta12":
        compute_cell_mass_func = compute_cell_mass_beta12
    elif trader_value_dist == "beta22":
        compute_cell_mass_func = compute_cell_mass_beta22
    else:
        raise ValueError(f"Unsupported trader_value_dist value: {trader_value_dist}")

    print("N:", N)
    mass_array = np.zeros((N + 1, N + 1))
    # Loop over cells (i,j) for i,j = 0,...,N-1.
    for i in range(N):
        for j in range(N):
            mass_array[i, j] = compute_cell_mass_func(i, j, N, c1, c2)
    # (Grid points corresponding to i == N or j == N get mass 0 by our construction.)

    # Now split the signed mass into positive (source) and negative (target) parts.
    S = np.maximum(mass_array, 0)
    T = np.maximum(-mass_array, 0)

    # Flatten the arrays in row-major order to match the grid_points ordering.
    S_flat = S.flatten()
    T_flat = T.flatten()
    return S_flat, T_flat


if __name__ == "__main__":
    N = 1001

    # Uniform trader value distribution, c=(1/3, 1/3), lambda=1
    c = np.array([1 / 3, 1 / 3])
    lambda_ = 1
    trader_value_dist = "uniform"

    # Beta(1, 2) trader value distribution, c=(1/2, 1/2), lambda=1
    # c = np.array([0.5, 0.5])
    # lambda_ = 1
    # trader_value_dist = "beta12"

    # Beta(2, 2) trader value distribution, c=(1/2, 1/2), lambda=1
    # c = np.array([0.5, 0.5])
    # lambda_ = 1
    # trader_value_dist = "beta22"

    optimal_transport_lp(lambda_, c, N, trader_value_dist)
