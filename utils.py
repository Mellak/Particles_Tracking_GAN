import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

def plot_paths_on2D(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, min_y=-6, max_y=6, radius=0.5):
    fig, ax = plt.subplots()  # Adjusted for 2D plotting
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[:stop]
        x = nonzero_a[:, 1]  # X-coordinate
        y = nonzero_a[:, 2]  # Y-coordinate
        ax.plot(x, y, color='red', label='Generated' if idx == 0 else "")

    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[:stop]
        x = nonzero_a[:, 1]  # X-coordinate
        y = nonzero_a[:, 2]  # Y-coordinate
        ax.plot(x, y, color='green', label='Real' if idx == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.legend()

    if save:
        plt.savefig(path)
    else:
        plt.show()

def plot_sphere_points(points_limit, points, paths=None, r=0.5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if points is not None:
        if isinstance(points, torch.Tensor):
            points = points.cpu().detach().numpy()

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ax.plot(x, y, z, color='red',marker='o', linewidth=2,)

    if points_limit is not None:
        x = points_limit[:, 0]
        y = points_limit[:, 1]
        z = points_limit[:, 2]
        ax.scatter(x, y, z, c='green')

    if paths is not None:
        print('here', paths.shape)
        nonzero_a = paths[np.any(paths != [0, 0, 0, 0], axis=1)]
        print(nonzero_a)
        print('here', nonzero_a.shape)
        x = nonzero_a[:, 1]
        y = nonzero_a[:, 2]
        z = nonzero_a[:, 3]
        #ax.scatter(x, y, z)
        ax.plot(x, y, z, color='orange')
    #ax.plot(x, y, z, color='red')
    #ax.scatter(x, y, z, color='red')

    # Define spherical coordinates
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r*np.outer(np.cos(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_generated_paths(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, min_y=-6, max_y=6, radius=0.5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[:stop]
        x = nonzero_a[:, 1]
        y = nonzero_a[:, 2]
        z = nonzero_a[:, 3]
        #ax.scatter(x, y, z)
        ax.plot(x, y, z, color='red')
    
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[:stop]
        x = nonzero_a[:, 1]
        y = nonzero_a[:, 2]
        z = nonzero_a[:, 3]
        #ax.scatter(x, y, z)
        ax.plot(x, y, z, color='green')

    

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    if save:
        plt.savefig(path)
    else:
        plt.show()

def plot_final_points(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, min_y=-6, max_y=6):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop]
        x = nonzero_a[:, 1]
        y = nonzero_a[:, 2]
        z = nonzero_a[:, 3]
        ax.scatter(x, y, z, color='red')
        #ax.plot(x, y, z, color='red')

    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop]
        x = nonzero_a[:, 1]
        y = nonzero_a[:, 2]
        z = nonzero_a[:, 3]
        ax.scatter(x, y, z, color='green')
        #ax.plot(x, y, z, color='green')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    if save:
        plt.savefig(path)
    else:
        plt.show()

def drawPSF(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, min_y=-6, max_y=6):
    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    points2 = np.empty((0,3))
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop,1:]
        points2 = np.append(points2, nonzero_a, axis=0)

    x_range = np.arange(min_x, max_x, 0.1)
    y_range = np.arange(min_y, max_y, 0.1)

    # Create a meshgrid of x and y values
    xx, yy = np.meshgrid(x_range, y_range)

    # Calculate the PSF for fake_path_test
    psf1 = np.zeros_like(xx)
    for p in points1:
        psf1 += np.exp(-((xx-p[0])**2 + (yy-p[1])**2)/(p[2]+0)**2)

    # Calculate the PSF for real_path
    psf2 = np.zeros_like(xx)
    for p in points2:
        psf2 += np.exp(-((xx-p[0])**2 + (yy-p[1])**2)/(p[2]+0)**2)

    # Plot the PSF for fake_path_test
    plt.subplot(121)
    plt.imshow(psf1, cmap='hot', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], vmax=2400,)
    plt.colorbar()
    plt.title('Fake PSF')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot the PSF for real_path
    plt.subplot(122)
    plt.imshow(psf2, cmap='hot', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], vmax=2400,)
    plt.colorbar()
    plt.title('Real PSF')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplots_adjust(wspace=0.4)
    if save:
        plt.savefig(path)
    else:
        plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def draw3DPSF(fake_path_test, real_path):
    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    points2 = np.empty((0,3))
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points2 = np.append(points2, nonzero_a, axis=0)
    x_range = np.arange(-4, 4, 0.1)
    y_range = np.arange(-4, 4, 0.1)

    # Create a meshgrid of x and y values
    xx, yy = np.meshgrid(x_range, y_range)

    # Calculate the PSF for fake_path_test
    psf1 = np.zeros_like(xx)
    for p in points1:
        psf1 += np.exp(-((xx-p[0])**2 + (yy-p[1])**2)/(p[2]+0)**2)

    # Calculate the PSF for real_path
    psf2 = np.zeros_like(xx)
    for p in points2:
        psf2 += np.exp(-((xx-p[0])**2 + (yy-p[1])**2)/(p[2]+0)**2)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface for fake_path_test
    surf1 = ax.plot_surface(xx, yy, psf1, cmap='Reds')

    # Plot the surface for real_path
    surf2 = ax.plot_surface(xx, yy, psf2, cmap='Greens')

    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('PSF')
    ax.set_title('PSF Comparison')

    # Add the colorbar
    #fig.colorbar(surf1)
    plt.show()

    # Get the dimensions of the PSF
    ny, nx = psf1.shape

    # Take a horizontal slice through the center of the PSF
    y = ny // 2
    profile1 = psf1[y, :]
    profile2 = psf2[y, :]

    # Plot the profiles with colored area underneath
    plt.fill_between(np.arange(nx), profile1, alpha=0.3, color='red')
    plt.plot(profile1, color='red', label='Fake PSF')
    plt.fill_between(np.arange(nx), profile2, alpha=0.3, color='green')
    plt.plot(profile2, color='green', label='Real PSF')
    plt.xlabel('X')
    plt.ylabel('Intensity')
    plt.title('Line Profiles through PSFs')
    plt.legend()
    plt.show()


def getUsefulPath(event):
    all_zero_rows = event[(event == 0).all(axis=1)]
    selected_rows = event[:-len(all_zero_rows)]
    return selected_rows

def make_values_zero(generated_data, data_count):
    for idx, event in enumerate(generated_data):
        limit = data_count[idx]
        generated_data[idx,limit:] = 0
        generated_data[idx,limit-1,0] = 0
    output = generated_data
    return output


def plot_x_distribution(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, num_bins=100):
    fig, ax = plt.subplots()

    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    points2 = np.empty((0,3))
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points2 = np.append(points2, nonzero_a, axis=0)

    bins = np.linspace(min_x, max_x, num_bins) #np.linspace(-6, 6, 0.1)

    # plot histogram of real points with green color
    x_real = points2[:, 0]  # extract x-coordinates of real points
    ax.hist(x_real, bins=bins, color='green', alpha=0.5, label='Real Points')

    # plot histogram of fake points with red color
    x_fake = points1[:, 0]  # extract x-coordinates of fake points
    ax.hist(x_fake, bins=bins, color='red', alpha=0.5, label='Fake Points')

    # set x and y axis labels and legend
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')

    if save:
        plt.savefig(path)
    else:
        plt.show()


def plot_y_distribution(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, num_bins=100):
    fig, ax = plt.subplots()

    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    points2 = np.empty((0,3))
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points2 = np.append(points2, nonzero_a, axis=0)

    bins = np.linspace(min_x, max_x, num_bins) #np.linspace(-6, 6, 0.1)

    # plot histogram of real points with green color
    x_real = points2[:, 1]  # extract x-coordinates of real points
    ax.hist(x_real, bins=bins, color='green', alpha=0.5, label='Real Points', )

    # plot histogram of fake points with red color
    x_fake = points1[:, 1]  # extract x-coordinates of fake points
    ax.hist(x_fake, bins=bins, color='red', alpha=0.5, label='Fake Points')

    # set x and y axis labels and legend
    ax.set_xlabel('Y-coordinate')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')

    if save:
        plt.savefig(path)
    else:
        plt.show()


def plot_z_distribution(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, num_bins=100):
    fig, ax = plt.subplots()

    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    points2 = np.empty((0,3))
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event)) #data_count[idx]
        nonzero_a = selected_event[stop-1:stop,1:]
        points2 = np.append(points2, nonzero_a, axis=0)

    bins = np.linspace(min_x, max_x, num_bins) #np.linspace(-6, 6, 0.1)
    # plot histogram of real points with green color
    x_real = points2[:, 2]  # extract x-coordinates of real points
    ax.hist(x_real, bins=bins, color='green', alpha=0.5, label='Real Points')

    # plot histogram of fake points with red color
    x_fake = points1[:, 2]  # extract x-coordinates of fake points
    ax.hist(x_fake, bins=bins, color='red', alpha=0.5, label='Fake Points')

    # set x and y axis labels and legend
    ax.set_xlabel('Z-coordinate')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')

    if save:
        plt.savefig(path)
    else:
        plt.show()

from matplotlib.widgets import Slider

from matplotlib.animation import FuncAnimation

def draw_animated(generated_grid, real_grid):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # Initialize the plots with the first slice
    im0 = ax[0].imshow(generated_grid[0], cmap='hot', vmin=0, vmax=100)
    ax[0].set_title('Generated 0')
    im1 = ax[1].imshow(real_grid[0], cmap='hot', vmin=0, vmax=100)
    ax[1].set_title('Real 0')
    im2 = ax[2].imshow(generated_grid[0] - real_grid[0], cmap='bwr', vmin=-10, vmax=10)
    ax[2].set_title('Diff 0')

    def update(slice):
        im0.set_data(generated_grid[slice])
        im1.set_data(real_grid[slice])
        im2.set_data(generated_grid[slice] - real_grid[slice])
        ax[0].set_title('Generated ' + str(slice))
        ax[1].set_title('Real ' + str(slice))
        ax[2].set_title('Diff ' + str(slice))

    ani = FuncAnimation(fig, update, frames=range(generated_grid.shape[0]), interval=200)
    plt.show()

def drawPSFwithDiff(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, min_y=-6, max_y=6, radius= 0.3):
    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    points2 = np.empty((0,3))
    for idx, selected_event in enumerate(real_path):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop,1:]
        points2 = np.append(points2, nonzero_a, axis=0)

    x_range = np.arange(min_x, max_x, 0.1)
    y_range = np.arange(min_y, max_y, 0.1)

    # Create a meshgrid of x and y values
    xx, yy = np.meshgrid(x_range, y_range)

    # Calculate the PSF for fake_path_test
    psf1 = np.zeros_like(xx)
    for p in points1:
        psf1 += np.exp(-((xx-p[0])**2 + (yy-p[1])**2)/(p[2]+0)**2)

    # Calculate the PSF for real_path
    psf2 = np.zeros_like(xx)
    for p in points2:
        psf2 += np.exp(-((xx-p[0])**2 + (yy-p[1])**2)/(p[2]+0)**2)

    # Create a function to update the PSF plots with a given vmax value
    def update_vmax(val):
        vmax_value = vmax_slider.val
        fake_plot.set_clim(vmax=vmax_value)
        real_plot.set_clim(vmax=vmax_value)
        plt.draw()

    # Create the figure
    fig = plt.figure(figsize=(12, 4))

    # Create subplots for PSF
    ax1 = plt.subplot(131)
    fake_plot = plt.imshow(psf1, cmap='hot', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], vmax=200,)
    plt.colorbar()
    plt.title('Fake PSF')
    plt.xlabel('X')
    plt.ylabel('Y')

    ax2 = plt.subplot(132)
    real_plot = plt.imshow(psf2, cmap='hot', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], vmax=200,)
    plt.colorbar()
    plt.title('Real PSF')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Draw horizontal and vertical lines at -radius and +radius
    ax1.axhline(-radius, color='red', linestyle='--', linewidth=1)
    ax1.axhline(radius, color='red', linestyle='--', linewidth=1)
    ax1.axvline(-radius, color='blue', linestyle='--', linewidth=1)
    ax1.axvline(radius, color='blue', linestyle='--', linewidth=1)

    ax2.axhline(-radius, color='red', linestyle='--', linewidth=1)
    ax2.axhline(radius, color='red', linestyle='--', linewidth=1)
    ax2.axvline(-radius, color='blue', linestyle='--', linewidth=1)
    ax2.axvline(radius, color='blue', linestyle='--', linewidth=1)

    # Create a slider axes on the right
    slider_ax = plt.axes([0.92, 0.2, 0.03, 0.6])
    vmax_slider = Slider(slider_ax, 'Vmax', 0, 10000, valinit=200, orientation="vertical")

    # Attach the slider's update function
    vmax_slider.on_changed(update_vmax)

    # Plot the PSF difference
    ax3 = plt.subplot(133)
    plt.imshow(psf2 - psf1, cmap='bwr', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], vmin=-300, vmax=300,)
    plt.colorbar()
    plt.title('Diff PSF')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplots_adjust(wspace=0.4)

    plt.show()


def discretize_coordinates(coordinates, image_shape=(15, 15, 15), voxel_size_mm=0.21):
    # Create an empty image
    image = np.zeros(image_shape)

    # Define the center voxel coordinates
    center_voxel = np.array(image_shape) // 2

    # Calculate the displacement to center the coordinates in the image
    displacement = center_voxel * voxel_size_mm

    # Convert the 3D coordinates to voxel indices
    voxel_indices = ((coordinates + displacement) / voxel_size_mm).astype(int)

    # Clip voxel indices to be within the image bounds
    voxel_indices = np.clip(voxel_indices, 0, np.array(image_shape) - 1)

    # Fill the image at the calculated voxel indices
    for voxel_index in voxel_indices:
        image[tuple(voxel_index)] += 1  # You can use any value you prefer

    return image

def generate_3D_images(fake_path_test, real_path, image_shape=(15, 15, 15), voxel_size_mm=0.21):
    points1 =  np.empty((0,3))
    for idx, selected_event in enumerate(fake_path_test):
        if isinstance(selected_event, torch.Tensor):
            selected_event = selected_event.cpu().detach().numpy()
        stop = len(getUsefulPath(selected_event))
        nonzero_a = selected_event[stop-1:stop,1:]
        points1 = np.append(points1, nonzero_a, axis=0)

    # check if real_path is None, then return only generated image
    if real_path is not None:
        points2 = np.empty((0,3))
        for idx, selected_event in enumerate(real_path):
            if isinstance(selected_event, torch.Tensor):
                selected_event = selected_event.cpu().detach().numpy()
            stop = len(getUsefulPath(selected_event))
            nonzero_a = selected_event[stop-1:stop,1:]
            points2 = np.append(points2, nonzero_a, axis=0)
        # Initialize an empty voxel grid
    generated_grid = discretize_coordinates(points1, image_shape=image_shape, voxel_size_mm=voxel_size_mm)
    if real_path is not None:
        real_grid = discretize_coordinates(points2, image_shape=image_shape, voxel_size_mm=voxel_size_mm)
    else:
        real_grid = None
    return generated_grid, real_grid


def get_final_points(fake_path_test, real_path):
    points1 =  np.empty((0,4))

    if isinstance(fake_path_test, torch.Tensor):
            fake_path_test = fake_path_test.cpu().detach().numpy()

    if isinstance(real_path, torch.Tensor):
            real_path = real_path.cpu().detach().numpy()

    for idx, selected_event in enumerate(fake_path_test):
        stop = len(getUsefulPath(selected_event))
        _point = selected_event[stop-1].reshape(1,4)
        points1 = np.append(points1, _point, axis=0)

    points2 = np.empty((0,4))
    for idx, selected_event in enumerate(real_path):
        stop = len(getUsefulPath(selected_event))
        _point = selected_event[stop-1].reshape(1,4)
        points2 = np.append(points2, _point, axis=0)
    return points1, points2


def plot_xyz_histograms(fake_path_test, real_path, save=False, path=None, min_x=-6, max_x=6, num_bins=100):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # Extract x, y, and z coordinates from fake points
    fake_points = np.concatenate([getUsefulPath(selected_event)[-1:, 1:] for selected_event in fake_path_test], axis=0)

    # Extract x, y, and z coordinates from real points
    real_points = np.concatenate([getUsefulPath(selected_event)[-1:, 1:] for selected_event in real_path], axis=0)

    # Define bins for the histograms
    bins = np.linspace(min_x, max_x, num_bins) 

    # Plot histogram of x-coordinates
    axes[0].hist(real_points[:, 0], bins=bins, color='green', alpha=0.5, label='Real Points')
    axes[0].hist(fake_points[:, 0], bins=bins, color='red', alpha=0.5, label='Fake Points')
    axes[0].set_xlabel('X-axis (mm)')
    axes[0].set_ylabel('Counts')
    axes[0].legend()

    # Plot histogram of y-coordinates
    axes[1].hist(real_points[:, 1], bins=bins, color='green', alpha=0.5, label='Real Points')
    axes[1].hist(fake_points[:, 1], bins=bins, color='red', alpha=0.5, label='Fake Points')
    axes[1].set_xlabel('Y-axis (mm)')
    axes[1].set_ylabel('Counts')
    axes[1].legend()

    # Plot histogram of z-coordinates
    axes[2].hist(real_points[:, 2], bins=bins, color='green', alpha=0.5, label='Real Points')
    axes[2].hist(fake_points[:, 2], bins=bins, color='red', alpha=0.5, label='Fake Points')
    axes[2].set_xlabel('Z-axis (mm)')
    axes[2].set_ylabel('Counts')
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if save:
        plt.savefig(path, bbox_inches='tight', dpi=1000)
    else:
        plt.show()