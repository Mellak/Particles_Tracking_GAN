# Standard library imports
import os
import numpy as np
import matplotlib.pyplot as plt

# PyTorch related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Function to rotate points around the center
def rotate_points_around_center(points, phi_degrees, theta_degrees):
    phi_rad = np.radians(phi_degrees)
    theta_rad = np.radians(theta_degrees)

    R_phi = np.array([[np.cos(phi_rad), -np.sin(phi_rad), 0],
                      [np.sin(phi_rad), np.cos(phi_rad), 0],
                      [0, 0, 1]])

    R_theta = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                        [0, 1, 0],
                        [-np.sin(theta_rad), 0, np.cos(theta_rad)]])

    R_combined = np.dot(R_phi, R_theta)
    rotated_points = np.dot(points, R_combined.T)

    return rotated_points

# Dataset class
class DataSetNumIntEnergyDirection(Dataset):
    def __init__(self, data_list, num_steps=18, num_features=4):
        input_tensor = np.concatenate(data_list)
        self.data = np.zeros((input_tensor.shape[0], num_steps, num_features))
        self.masks = np.zeros((input_tensor.shape[0], num_steps))

        for j, event in enumerate(input_tensor):
            all_zero_rows = event[(event == 0).all(axis=1)]
            selected_rows = event[:-len(all_zero_rows)]
            starting_point = selected_rows[0, 1:]

            selected_rows[:, 1:] = selected_rows[:, 1:] - starting_point
            self.data[j, :-len(all_zero_rows), :] = selected_rows

        self.Energies = self.data[:, 0, 0]
        self.numInteractions = num_steps - np.count_nonzero(np.count_nonzero(self.data, axis=2) == 0, axis=1)
        self.masks = ~(torch.arange(num_steps).expand(len(self.numInteractions), -1) >= torch.tensor(self.numInteractions).view(-1, 1))
        
        
        self.data = np.transpose(self.data, (0, 2, 1))
        self.data = self.data[:, :, np.newaxis, :]
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data[idx].squeeze().transpose(1, 0)
        end_idx = self.num_steps - np.count_nonzero(np.count_nonzero(data_tensor, axis=1) == 0, axis=0)

        angle2 = np.random.randint(0, 360)
        angle3 = np.random.randint(0, 360)

        rotated_points = rotate_points_around_center(data_tensor[:end_idx, 1:4], angle2, angle3)
        rotated_points = np.expand_dims(rotated_points.T, axis=1)
        f_rotated_points = np.copy(self.data[idx])
        f_rotated_points[1:4, :, :end_idx] = torch.from_numpy(rotated_points)

        _f_rotated_points = np.copy(f_rotated_points)[:, 0, :].transpose(1, 0)
        points1 = _f_rotated_points[0, 1:4]
        points2 = _f_rotated_points[1, 1:4]
        # normalized vector between the points1 and points2
        normalized_vector = (points2 - points1) / np.linalg.norm(points2 - points1)


        return f_rotated_points, self.numInteractions[idx], self.Energies[idx], self.masks[idx], normalized_vector

class FastDataloader(Dataset):
    def __init__(self, input_tensor, num_steps=18, num_features=4):
        self.num_steps = num_steps
        self.num_features = num_features
        #input_tensor = np.concatenate(data_list)
        self.data = np.zeros((input_tensor.shape[0], self.num_steps, self.num_features))
        self.masks = np.zeros((input_tensor.shape[0], self.num_steps))
        self.normalized_vector = np.zeros((input_tensor.shape[0], 3))

        for j, event in enumerate(input_tensor):
            all_zero_rows = event[(event == 0).all(axis=1)]
            selected_rows = event[:-len(all_zero_rows)]
            starting_point = selected_rows[0, 1:]

            selected_rows[:, 1:] = selected_rows[:, 1:] - starting_point
            self.data[j, :-len(all_zero_rows), :] = selected_rows

        
            points1 = self.data[j, 0, 1:4]
            points2 = self.data[j, 1, 1:4]
            # normalized vector between the points1 and points2
            normalized_vector = (points2 - points1) / np.linalg.norm(points2 - points1)
            self.normalized_vector[j] = normalized_vector

            # Those are outliers, we should remove them
            _distance = np.linalg.norm(selected_rows[-1, 1:] - selected_rows[0, 1:])
            if _distance>100:
                continue

        self.Energies = self.data[:, 0, 0]
        self.numInteractions = num_steps - np.count_nonzero(np.count_nonzero(self.data, axis=2) == 0, axis=1)
        self.masks = ~(torch.arange(num_steps).expand(len(self.numInteractions), -1) >= torch.tensor(self.numInteractions).view(-1, 1))
        
        
        self.data = np.transpose(self.data, (0, 2, 1))
        self.data = self.data[:, :, np.newaxis, :]
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.numInteractions[idx], self.Energies[idx], self.masks[idx], self.normalized_vector[idx]