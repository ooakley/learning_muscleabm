import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib
    import cv2
    import scipy

    import numpy as np
    import pandas as pd
    import colorcet as cc

    import matplotlib.pyplot as plt
    return cc, np, os, plt


@app.cell
def _():
    import awkward as ak
    return (ak,)


@app.cell
def _(np):
    # Data location:
    DATA_DIR_PATH = "configs/model_parameter_json/matrix_test"

    # Matrix variables:
    MESH_NUMBER = 64
    HEXAGONAL_ROWS = np.floor(MESH_NUMBER / (np.sqrt(3) / 2)).astype(int)
    return DATA_DIR_PATH, HEXAGONAL_ROWS, MESH_NUMBER


@app.cell
def _(DATA_DIR_PATH, np, os):
    def read_matrix_into_numpy(subiteration):
        # Determine filepaths:
        filename = f"matrix_seed{subiteration:03d}.txt"
        filepath = os.path.join(DATA_DIR_PATH, filename)

        array_list = []
        with open(filepath, "r") as f:
            for line in f:
                heading_string = str(line.rstrip())
                array_list.append(np.fromstring(heading_string, sep=","))

        return array_list
    return (read_matrix_into_numpy,)


@app.cell
def _(read_matrix_into_numpy):
    array_list = read_matrix_into_numpy(0)
    return (array_list,)


@app.cell
def _(array_list, np):
    np.concatenate(array_list).min()
    return


@app.cell
def _(array_list, np):
    np.concatenate(array_list).max()
    return


@app.cell
def _(MESH_NUMBER, array_list, np):
    def extract_heading_components(array_list):
        heading_components = np.zeros((MESH_NUMBER**2, 3))

        for i in range(MESH_NUMBER**2):
            # Decompose headings:
            for heading in array_list[i]:
                shifted_heading = heading * 2
                heading_components[i, 0] += np.clip(np.cos(shifted_heading), a_min=0, a_max=None)
                heading_components[i, 1] += np.clip(np.cos(shifted_heading - (np.pi * (2 / 3))), a_min=0, a_max=None)
                heading_components[i, 2] += np.clip(np.cos(shifted_heading - (np.pi * (4 / 3))), a_min=0, a_max=None)

        return heading_components.reshape((MESH_NUMBER, MESH_NUMBER, -1))

    heading_components = extract_heading_components(array_list)
    return (heading_components,)


@app.cell
def _(heading_components, plt):
    def plot_components(heading_components):
        fig, axs = plt.subplots(1, 3, figsize=(8, 5))
        for i in range(3):
            axs[i].imshow(heading_components[:, :, i], vmin=0, vmax=None, origin="upper")
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        fig.tight_layout()
        plt.show()

    plot_components(heading_components)
    return


@app.cell
def _(MESH_NUMBER, heading_components, np):
    def generate_hexagonal_components(heading_components):
        # https://stackoverflow.com/questions/39421233/convert-image-pixels-from-square-to-hexagonal
        # Basically, we need to think of the arrangement of points that generates a triangular meshwork.
        # Each row will only by sqrt(3) / 2 (approx. 0.82) below the previous row, meaning we will need
        # more rows of dots than rows of pixels to accurately represent the matrix.

        # Intermediate rows progress by this much (approx 0.86):
        hexagonal_rows = np.floor(MESH_NUMBER / (np.sqrt(3) / 2)).astype(int)

        # Generate grid object:
        hexagonal_grid_object = []
        for hex_index in range(hexagonal_rows):
            # Get distance relative to top row:
            row_distance = 0.5 + (hex_index * np.sqrt(3) / 2)
            matrix_index = int(np.floor(row_distance))
            matrix_offset = row_distance % 1.0

            # Squeeze even-numbered rows:
            if (hex_index % 2) == 0:
                # Interpolate across row:
                component_differences = np.diff(heading_components[matrix_index, :, :], axis=0)
                interpolated_components = heading_components[matrix_index, :-1, :] + (component_differences / 2)

                # If below row midline, interpolate from upper row:
                if matrix_offset <= 0.5:
                    # Get relative proportion:
                    upper_proportion = 0.5 - matrix_offset
                    upper_component_differences = \
                        np.diff(heading_components[matrix_index - 1, :, :], axis=0)
                    upper_interpolated_components = \
                        heading_components[matrix_index - 1, :-1, :] \
                        + (upper_component_differences / 2)
                    hexagonal_grid_object.append(
                        (upper_proportion * upper_interpolated_components) + 
                        ((1 - upper_proportion) * interpolated_components)
                    )

                # If above row midline, interpolate from lower row:
                else:
                    # Get relative proportion:
                    lower_proportion = matrix_offset - 0.5
                    lower_component_differences = \
                        np.diff(heading_components[matrix_index + 1, :, :], axis=0)
                    lower_interpolated_components = \
                        heading_components[matrix_index + 1, :-1, :] \
                        + (lower_component_differences / 2)
                    hexagonal_grid_object.append(
                        (lower_proportion * lower_interpolated_components) + 
                        ((1 - lower_proportion) * interpolated_components)
                    )

            # Retain normal cardinality for odd-numbered rows:
            else:
                # If below row midline, interpolate from upper row:
                if matrix_offset <= 0.5:
                    # Get relative proportion:
                    upper_proportion = 0.5 - matrix_offset
                    hexagonal_grid_object.append(
                        (upper_proportion * heading_components[matrix_index - 1, :, :]) + 
                        ((1 - upper_proportion) * heading_components[matrix_index, :, :])
                    )
                # If above row midline, interpolate from lower row:
                else:
                    # Get relative proportion:
                    lower_proportion = matrix_offset - 0.5
                    hexagonal_grid_object.append(
                        (lower_proportion * heading_components[matrix_index + 1, :, :]) + 
                        ((1 - lower_proportion) * heading_components[matrix_index, :, :])
                    )

        return hexagonal_grid_object

    hexagonal_matrix_components = generate_hexagonal_components(heading_components)
    return (hexagonal_matrix_components,)


@app.cell
def _(hexagonal_matrix_components, np):
    hex_array = np.concatenate(hexagonal_matrix_components, axis=0)
    return (hex_array,)


@app.cell
def _():
    EVEN_CONNECTIVITY = [
        # Along 0 degrees:
        [0, 1],
        [0, -1],

        # Along 60 degrees:
        [-1, 1],
        [1, 0],

        # Along 120 degrees:
        [-1, 0],
        [1, 1]
    ]

    ODD_CONNECTIVITY = [
        # Along 0 degrees:
        [0, 1],
        [0, -1],

        # Along 60 degrees:
        [-1, 0],
        [1, -1],

        # Along 120 degrees:
        [-1, -1],
        [1, 0]
    ]
    return EVEN_CONNECTIVITY, ODD_CONNECTIVITY


@app.cell
def _(ak, hex_array, hexagonal_matrix_components):
    component_densities = ak.Array(hexagonal_matrix_components)
    node_count = hex_array.shape[0]
    return (node_count,)


@app.cell
def _(MESH_NUMBER, np):
    def get_absolute_index(i, j):
        even_rows = int(np.ceil(i / 2))
        odd_rows = int(np.floor(i / 2))
        elapsed_row_nodes = (even_rows * (MESH_NUMBER - 1)) + (odd_rows * (MESH_NUMBER))
        return elapsed_row_nodes + j
    return (get_absolute_index,)


@app.cell
def _(HEXAGONAL_ROWS, MESH_NUMBER):
    def is_node_in_bounds(i, j):
        row_in_bounds = 0 <= i < HEXAGONAL_ROWS
        if (i % 2) == 0:
            column_in_bounds = 0 <= j < (MESH_NUMBER - 1)
        else:
            column_in_bounds = 0 <= j < MESH_NUMBER
        return row_in_bounds & column_in_bounds
    return (is_node_in_bounds,)


@app.cell
def _(
    EVEN_CONNECTIVITY,
    HEXAGONAL_ROWS,
    MESH_NUMBER,
    ODD_CONNECTIVITY,
    get_absolute_index,
    hex_array,
    is_node_in_bounds,
    node_count,
    np,
):
    # Generate spring constant (roughly) sparse matrix:
    def generate_spring_matrix():
        adjacency_matrix = np.zeros((node_count, node_count))
        k_matrix = np.zeros((node_count, node_count))
        previous_index = -1
        for i in range(HEXAGONAL_ROWS):
            if (i % 2) == 0:
                connectivity = EVEN_CONNECTIVITY
                nodes_in_row = MESH_NUMBER - 1
            else:
                connectivity = ODD_CONNECTIVITY
                nodes_in_row = MESH_NUMBER
            for j in range(nodes_in_row):
                current_index = get_absolute_index(i, j)
                assert (current_index - previous_index) == 1
                previous_index = current_index
                for angular_index, (i_diff, j_diff) in enumerate(connectivity):
                    if not is_node_in_bounds(i + i_diff, j + j_diff):
                        continue

                    # Set up adjacency matrix:
                    new_connected_index = get_absolute_index(i + i_diff, j + j_diff)
                    connected_index = new_connected_index
                    adjacency_matrix[current_index, connected_index] = 1

                    # Set up spring constant matrix:
                    component_index = int(np.floor(angular_index / 2))
                    averaged_k = (hex_array[current_index, component_index] + hex_array[connected_index, component_index]) / 2
                    k_matrix[current_index, connected_index] = averaged_k

        return adjacency_matrix, k_matrix

    adjacency_matrix, k_matrix = generate_spring_matrix()
    return adjacency_matrix, k_matrix


@app.cell
def _(np):
    def dist(X, Y):
        sx = np.sum(X**2, axis=1, keepdims=True)
        sy = np.sum(Y**2, axis=1, keepdims=True)
        return np.sqrt(-2 * X.dot(Y.T) + sx + sy.T)
    return


@app.cell
def _(HEXAGONAL_ROWS, MESH_NUMBER, np):
    def generate_node_positions():
        # Generate positions:
        positions_object = []

        # Get relative positions:
        reference_x = np.arange(MESH_NUMBER)

        # Iterate through rows:
        for row_index in range(HEXAGONAL_ROWS):
            # Account for interpolation from grid:
            if (row_index % 2) == 0:
                nodes_in_row = MESH_NUMBER - 1
            else:
                nodes_in_row = MESH_NUMBER

            # Iterate through nodes and assign positions:
            row_positions = np.empty((nodes_in_row, 2))
            for j_index in range(nodes_in_row):
                # Assign x:
                if (row_index % 2) == 0:
                    row_positions[j_index, 0] = reference_x[j_index] + 1.0
                else:
                    row_positions[j_index, 0] = reference_x[j_index] + 0.5

            # Assign y (for entire row):
            row_positions[:, 1] = MESH_NUMBER - (0.5 + (row_index * (np.sqrt(3) / 2)))

            # Accumulate to positions object:
            positions_object.append(row_positions)

        return positions_object

    positions_object = generate_node_positions()
    positions_array = np.concatenate(positions_object, axis=0)
    return positions_array, positions_object


@app.cell
def _(positions_array):
    positions_array
    return


@app.cell
def _(adjacency_matrix, copy, k_matrix, node_count, np):
    def vectorised_loop(initial_positions):
        # Initialise positions:
        positions = copy.deepcopy(initial_positions)

        # Set up indices for connected points:
        distance_comparisons = np.stack(np.where(adjacency_matrix), axis=1)
        matrix_indices = np.nonzero(adjacency_matrix)
        # Initialise data matrices:
        displacement_matrix = np.zeros((node_count, node_count, 2))
        force_matrix = np.zeros((node_count, node_count))

        # Spring constant array:
        k_array = k_matrix[np.nonzero(k_matrix)].flatten()

        # Iterate through updates:
        for i in range(500):
            # Get displacements:
            displacements = positions[distance_comparisons[:, 1], :] - positions[distance_comparisons[:, 0], :]

            # Infer forces from extensions and k matrix:
            distances = np.sqrt(np.sum(displacements ** 2, axis=1))
            extensions = np.clip(distances - 1.0, 0, None)

            # Get forces acting on points:
            displacement_matrix[matrix_indices] = displacements
            force_matrix[matrix_indices] = extensions * k_array * 1
            force_array = np.sum(displacement_matrix * np.expand_dims(force_matrix, axis=2), axis=1)
            force_array[:63, 1] += 20
            force_array[-63:, :] = 0
            positions += force_array * 0.005

        return positions
    return (vectorised_loop,)


@app.cell
def _(adjacency_matrix, k_matrix, node_count, np):
    import torch

    class ImplicitSolverPytorch:
        def __init__(self, initial_positions):
            # Initialise positions:
            self.positions = torch.from_numpy(initial_positions)

            # Set up indices for connected points:
            self.distance_comparisons = torch.from_numpy(np.stack(np.where(adjacency_matrix), axis=1))
            self.matrix_indices = np.nonzero(adjacency_matrix)

            # Initialise data matrices:
            self.displacement_matrix = torch.from_numpy(np.zeros((node_count, node_count, 2)))
            self.force_matrix = torch.from_numpy(np.zeros((node_count, node_count)))
    
            # Spring constant array:
            self.k_array = torch.from_numpy(k_matrix[np.nonzero(k_matrix)].flatten())

        def get_velocities(self, positions):
            # Get displacements:
            displacements = \
                positions[self.distance_comparisons[:, 1], :] \
                - positions[self.distance_comparisons[:, 0], :]

            # Get spring extensions:
            distances = torch.sqrt(torch.sum(displacements ** 2, dim=1))
            extensions = torch.clip(distances - 1.0, 0, None)

            # Get forces acting on points:
            self.displacement_matrix[self.matrix_indices] = displacements
            self.force_matrix[self.matrix_indices] = extensions * self.k_array * 0.1
            force_array = torch.sum(self.displacement_matrix * torch.unsqueeze(self.force_matrix, dim=2), dim=1)
            force_array[:63, 1] += 20

            # Define level of damping in our system:
            return 0.1 * force_array

        def calculate_step(self, h):
            # Instantiate next positions (initial guess with forward Euler):
            next_positions = self.positions + (h * self.get_velocities(self.positions))
            next_positions.requires_grad_(True)

            converged = False
            while not converged:
                # Calculate implicit function:
                forces = self.get_velocities(next_positions)
                zero = next_positions - self.positions - (h * forces)
                error = torch.sum(zero**2)
                print(error.item())
                grad_x, = torch.autograd.grad(error, next_positions)
                next_positions = next_positions - 0.01*grad_x
                if error < 1:
                    print("Converged!")
                    converged = True

            self.positions = next_positions.detach().requires_grad_(False)
    return (ImplicitSolverPytorch,)


@app.cell
def _(ImplicitSolverPytorch, positions_array):
    implicit_solver = ImplicitSolverPytorch(positions_array)
    return (implicit_solver,)


@app.cell
def _(implicit_solver, positions_array):
    implicit_solver.positions.detach() - positions_array
    return


@app.cell
def _(implicit_solver):
    implicit_solver.calculate_step(1)
    return


@app.cell
def _(positions_array, vectorised_loop):
    test_positions = vectorised_loop(positions_array)
    return


@app.cell
def _(hex_array, implicit_solver, plt):
    def plot_vectorised_positions(positions):
        fig, ax = plt.subplots()
        ax.scatter(positions[:, 0], positions[:, 1], s=1, c=hex_array[:, 1], clim=(0, None))
        ax.set_aspect("equal")
        # ax.set_xlim(-0.5, 65)
        # ax.set_ylim(0.5, 65)
        plt.show()

    plot_vectorised_positions(implicit_solver.positions.numpy())
    return


@app.cell
def _(HEXAGONAL_ROWS, cc, hexagonal_matrix_components, plt, positions_object):
    def plot_positions(ax, positions_object, component):
        for i in range(HEXAGONAL_ROWS):
            ax.scatter(
                positions_object[i][:, 0], positions_object[i][:, 1],
                c=hexagonal_matrix_components[i][:, component],
                s=0.7, cmap=cc.m_CET_L3
            )
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_facecolor('k')

    def plot_hex_components(positions_object):
        fig, axs = plt.subplots(1, 3, figsize=(12, 8))
        for i in range(3):
            plot_positions(axs[i], positions_object, i)
        fig.tight_layout()
        plt.show()

    plot_hex_components(positions_object)
    return (plot_hex_components,)


@app.cell
def _(EVEN_CONNECTIVITY, HEXAGONAL_ROWS, MESH_NUMBER, ODD_CONNECTIVITY, np):
    import copy

    RESTING_LENGTH = 1.0
    B = 0.1
    D_T = 0.25
    BASE_SPRING_CONSTANT = 1.0

    class MatrixLatticeSpring:

        def __init__(self, hexagonal_matrix_components, initial_positions):
            # Instantiate internal representations of stiffness and node positions:
            self.t = 0
            self.matrix_components = hexagonal_matrix_components
            self.positions = copy.deepcopy(initial_positions)
            self.calculate_spring_constants()

        def nodes_in_row(self, row_index):
            # Account for interpolation from grid:
            if (row_index % 2) == 0:
                return MESH_NUMBER - 1
            else:
                return MESH_NUMBER

        def is_node_in_bounds(self, i, j):
            row_in_bounds = 0 <= i < HEXAGONAL_ROWS
            if (i % 2) == 0:
                column_in_bounds = 0 <= j < (MESH_NUMBER - 1)
            else:
                column_in_bounds = 0 <= j < MESH_NUMBER
            return row_in_bounds & column_in_bounds

        def add_node_to_spring_constant_map(self, i, j):
            # Ensure we have the correct relationship between nodes:
            assert self.is_node_in_bounds(i, j)
            if (i % 2) == 0:
                connectivity = EVEN_CONNECTIVITY
            else:
                connectivity = ODD_CONNECTIVITY

            for angular_index, (i_diff, j_diff) in enumerate(connectivity):
                # Skip if out-of-bounds:
                connected_i = i + i_diff
                connected_j = j + j_diff
                if not self.is_node_in_bounds(connected_i, connected_j):
                    continue

                # Get index of relevant angular component:
                component_index = int(np.floor(angular_index / 2))
                current_component = self.matrix_components[i][j, component_index]
                connected_component = \
                    self.matrix_components[connected_i][connected_j, component_index]
                joint_component = (current_component + connected_component) / 2
                joint_component *= BASE_SPRING_CONSTANT

                # Key to component with tuple of indices:
                self.spring_constant_map[(i, j, i_diff, j_diff)] = joint_component

        def calculate_spring_constants(self):
            # Instantiate dictionary:
            self.spring_constant_map = {}
            for i in range(HEXAGONAL_ROWS):
                for j in range(self.nodes_in_row(i)):
                    self.add_node_to_spring_constant_map(i, j)

        def calculate_force_at_node(self, i_index, j_index):
            # Set up comparison position:
            force = np.zeros(2)
            reference_xy = self.positions[i_index][j_index, :]
            if (i_index % 2) == 0:
                connectivity = EVEN_CONNECTIVITY
            else:
                connectivity = ODD_CONNECTIVITY

            for i_diff, j_diff in connectivity:
                # Skip if out-of-bounds:
                connected_i = i_index + i_diff
                connected_j = j_index + j_diff
                if not self.is_node_in_bounds(connected_i, connected_j):
                    continue

                # Calculate displacement to connection:
                connected_xy = self.positions[connected_i][connected_j, :]
                displacement = connected_xy - reference_xy
                distance = np.sqrt(np.sum((displacement)**2))
                # extension = np.clip(distance - RESTING_LENGTH, 0, None)
                extension = distance - RESTING_LENGTH
                relevant_constant = self.spring_constant_map[
                    (i_index, j_index, i_diff, j_diff)
                ]
                force_magnitude = extension * relevant_constant

                # Caculate force components:
                force += force_magnitude * displacement

            return force

        def update_spring_network(self, force=0):
            # Iterate via row:
            new_positions = []
            for i in range(HEXAGONAL_ROWS):
                # Iterate through nodes and assign forces:
                nodes_in_row = self.nodes_in_row(i)
                row_positions = np.empty((nodes_in_row, 2))
                for j in range(nodes_in_row):
                    if i == 0:
                        # Calculate local force:
                        force_at_node = self.calculate_force_at_node(i, j)

                        # Reaction force lateral to pulling surface:
                        force_at_node[0] = 0

                        # Force applied at pulling surface:
                        force_at_node[1] += force

                        # Apply corrected force to position:
                        current_position = self.positions[i][j, :]
                        velocity = B * force_at_node
                        row_positions[j, :] = current_position + (velocity * D_T)
                    else:
                        force_at_node = self.calculate_force_at_node(i, j)
                        current_position = self.positions[i][j, :]
                        velocity = B * force_at_node
                        row_positions[j, :] = current_position + (velocity * D_T)
                new_positions.append(row_positions)
            return new_positions

        def run_simulation_step(self):
            # Calculate next node positions:
            new_positions = self.update_spring_network(np.min([self.t*10, 10]))

            # Fix top and bottom rows in place:
            # new_positions[0] = self.positions[0]
            new_positions[-1] = self.positions[-1]

            # Update positions:
            self.positions = new_positions
            # if self.t < 200:
            #     self.positions[0][:, 1] += 0.001 * D_T

            # Update time:
            self.t += D_T

        def plot_connections(self, ax):
            # Iterate via row:
            scatter_data = []
            x_plot_data = []
            y_plot_data = []
            for i in range(HEXAGONAL_ROWS):
                nodes_in_row = self.nodes_in_row(i)
                row_positions = np.empty((nodes_in_row, 2))
                if (i % 2) == 0:
                    connectivity = EVEN_CONNECTIVITY
                else:
                    connectivity = ODD_CONNECTIVITY
                for j in range(nodes_in_row):
                    current_x = self.positions[i][j, 0]
                    current_y = self.positions[i][j, 1]
                    scatter_data.append(self.positions[i][j, :])
                    for i_diff, j_diff in connectivity:
                        # Skip if out-of-bounds:
                        connected_i = i + i_diff
                        connected_j = j + j_diff
                        if not self.is_node_in_bounds(connected_i, connected_j):
                            continue

                        # Plot:
                        connected_x = self.positions[connected_i][connected_j, 0]
                        connected_y = self.positions[connected_i][connected_j, 1]
                        x_plot_data.append([current_x, connected_x])
                        y_plot_data.append([current_y, connected_y])

            scatter_data = np.stack(scatter_data, axis=0)
            # ax.scatter(scatter_data[:, 0], scatter_data[:, 1], s=1)

            x_plot_data = np.stack(x_plot_data, axis=1)
            y_plot_data = np.stack(y_plot_data, axis=1)
            ax.plot(x_plot_data, y_plot_data, lw=1, c='k')
    return D_T, MatrixLatticeSpring, copy


@app.cell
def _(MatrixLatticeSpring, hexagonal_matrix_components, positions_object):
    lattice = MatrixLatticeSpring(hexagonal_matrix_components, positions_object)
    return (lattice,)


@app.cell
def _(lattice, plt):
    fig, ax = plt.subplots(figsize=(10, 10))
    lattice.plot_connections(ax)
    ax.set_aspect("equal")
    plt.show()
    return


@app.cell
def _(D_T, lattice):
    print(500 * D_T)
    for i in range(200):
        lattice.run_simulation_step()
    return


@app.cell
def _(lattice, plot_hex_components):
    plot_hex_components(lattice.positions)
    return


@app.cell
def _():
 
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
