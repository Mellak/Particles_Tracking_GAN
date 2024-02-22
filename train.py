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

# Local imports from custom modules
from models import GeneratorNumIntEnergyDirection2, ViTwMask2
from utils import generate_3D_images
from dataloader import DataSetNumIntEnergyDirection


# Define constants and configurations
POSITRON_EMITTER = "F18" # "F18" is an option
LATENT_DIM = 100
NUM_STEPS = 30 if POSITRON_EMITTER == "Ga62" else 18
NUM_FEATURES = 4  # Energy, X, Y, Z
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 30
MATERIAL = "Water"

DATA_DIR = f"/home/youness/data/Positron_Range_Project/WData/{MATERIAL}{POSITRON_EMITTER}/"
MIN_SELECTION = 0
MAX_SELECTION = 20
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000

# Paths for saving models and outputs
PATH_BASE    = f'/home/youness/data/Konstantinos_Model/Experiences{POSITRON_EMITTER}/{MATERIAL}/'
PATH_WEIGHTS = os.path.join(PATH_BASE, 'Weights', f'TransfGAN_{MATERIAL}/')
PATH_IMAGES  = os.path.join(PATH_BASE,  'Images', f'TransfGAN_{MATERIAL}/')
PATH_KERNELS = os.path.join(PATH_BASE, 'Kernels', f'TransfGAN_{MATERIAL}/')

for path in [PATH_WEIGHTS, PATH_IMAGES, PATH_KERNELS]:
    os.makedirs(path, exist_ok=True)

# Define loss functions
def Energy_loss(generated_data):
    below_zero_penalty = torch.relu(-generated_data).sum(dim=-1)
    diff = generated_data[:, :, 1:] - generated_data[:, :, :-1]
    non_decreasing_penalty = torch.relu(diff).sum(dim=-1)
    total_penalty = non_decreasing_penalty + below_zero_penalty
    total_penalty = 0.005 * total_penalty.mean()
    return total_penalty

Criterion_Angle = nn.CosineSimilarity(dim=1, eps=1e-6) 

def train_model_step(batch_size, discriminator, generator, d_optimizer, g_optimizer, real_paths, num_inter, energy, masks, start_vectors):
    
    d_optimizer.zero_grad()
   

    real_paths_d = real_paths #torch.cat((output, real_paths), dim=1)
    # Train discriminator
    real_validity = discriminator(real_paths_d, num_inter, energy, masks)
    

    z = Variable(torch.randn(batch_size, LATENT_DIM)).cuda()
    fake_num_inter = num_inter #[torch.randperm(num_inter.size(0))]  # Shuffle the num_inter
    fake_paths = generator(z, fake_num_inter, energy, masks, start_vectors)
    fake_paths_d = fake_paths #torch.cat((outputf, fake_paths), dim=1)
    fake_validity = discriminator(fake_paths_d.detach(), fake_num_inter, energy, masks)
    d_loss1 = nn.MSELoss()(real_validity, torch.ones_like(real_validity)) + nn.MSELoss()(fake_validity, torch.zeros_like(fake_validity))
    

    d_loss = d_loss1

    d_loss.backward()
    d_optimizer.step()


    g_optimizer.zero_grad()

    # Train generator
    validity  = discriminator(fake_paths_d, fake_num_inter, energy, masks)

    energy_column_loss = Energy_loss(fake_paths[:, 0, :, :])

    points1 = fake_paths[:, 1:4, 0, 0]
    points2 = fake_paths[:, 1:4, 0, 1]
    # normalized vector between the points1 and points2 with pytorch:
    normalized_vector = (points2 - points1) / torch.norm(points2 - points1, dim=1).unsqueeze(1)
    
    cosine_loss = 1e0 * torch.mean((1 - (Criterion_Angle(normalized_vector, start_vectors))))
    
    g_loss1 = nn.MSELoss()(validity, torch.ones_like(validity)) + energy_column_loss + cosine_loss

    g_loss = g_loss1 
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item(), d_loss.item()

# Main training loop
def main():
    # Load data, create dataset and dataloader
    # Load data and create dataset and dataloader
    Input_data_list = [np.load(os.path.join(DATA_DIR, f)) for f in sorted(os.listdir(DATA_DIR)) if ("positrons_" in f and int(f.replace("positrons_", "").replace(".npy", "")) < MAX_SELECTION) and int(f.replace("positrons_", "").replace(".npy", "")) > MIN_SELECTION]
    dataset     = DataSetNumIntEnergyDirection(Input_data_list, num_steps=NUM_STEPS, num_features=NUM_FEATURES)
    dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    tdataloader = DataLoader(dataset, batch_size=10000, shuffle=True)

    # Define the generator and discriminator
    generator      = GeneratorNumIntEnergyDirection2(seq_len=NUM_STEPS).to(DEVICE)
    discriminator  = ViTwMask2(image_size=NUM_STEPS, patch_size=1, num_classes=1, channels=NUM_FEATURES, dim=64, depth=3, heads=4, mlp_dim=128).to(DEVICE)

    # Initialize optimizers
    lr = 1e-4
    optimizer_G  = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_D  = torch.optim.Adam(discriminator.parameters(), lr=3*lr, betas=(0.9, 0.999))

    # Load checkpoint if exists
    Start_EPOCH = 0
    if os.path.isfile(PATH_WEIGHTS+"Training_TransfGAN_"+MATERIAL+".pth"):
        Start_EPOCH = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D) + 1
    else:
        print('----------Start from scratch------------')
    # Start training loop
    for epoch in range(Start_EPOCH, NUM_EPOCHS):
        print('Starting epoch {}...'.format(epoch), end=' ')
        step = 0

        for i, (real_path, num_inter, energy, masks, start_vector) in enumerate(dataloader):
            if i == len(dataloader) - 1 and num_inter.shape[0] < BATCH_SIZE:
                break
            step = epoch * len(dataloader) + i + 1

            real_path = Variable(real_path).type(torch.float32).cuda()
            num_inter = Variable(num_inter).type(torch.long).cuda()
            energy = Variable(energy).type(torch.float32).cuda()
            masks = Variable(masks).type(torch.float32).cuda()
            start_vector = Variable(start_vector).type(torch.float32).cuda()

            generator.train()
            g_loss, d_loss =  train_model_step(len(real_path), discriminator, generator, optimizer_D, optimizer_G, real_path, num_inter, energy, masks, start_vector)
            if i % 500 == 0:
                print('scalars', {'g_loss': g_loss, 'd_loss': (d_loss)}, step)
        if epoch % 1 == 0:
            with torch.no_grad():
                for i, batch in enumerate(tdataloader):
                    real_path, num_inter, energy, masks, start_vector = batch
                    break

                num_inter = num_inter.type(torch.long).to(DEVICE)#[:100]
                energy = energy.type(torch.float32).to(DEVICE)#[:100]
                masks = masks.type(torch.float32).to(DEVICE)#[:100]
                real_path = real_path.type(torch.float32).to(DEVICE)#[:100]
                start_vector = start_vector.type(torch.float32).to(DEVICE)#[:100]

                hidden = Variable(torch.randn(num_inter.shape[0], LATENT_DIM)).to(DEVICE)
                fake_path_test = generator(hidden, num_inter, energy, masks, start_vector).to(DEVICE)
                fake_path_test = fake_path_test * masks.unsqueeze(1).unsqueeze(1)

                fake_path_test = fake_path_test[:, :, 0, :]
                real_path = real_path[:, :, 0, :]

                fake_path_test = fake_path_test.transpose(1, 2)
                real_path = real_path.transpose(1, 2)

                generated_grid, real_grid = generate_3D_images(fake_path_test, real_path, image_shape=(15, 15, 15), voxel_size_mm=0.5)
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                ax[0].imshow(generated_grid[generated_grid.shape[0]//2, :, :], cmap ='hot')
                ax[0].set_title("Generated")
                ax[1].imshow(real_grid[generated_grid.shape[0]//2, :, :], cmap ='hot')
                ax[1].set_title("real")
                ax[2].imshow(generated_grid[generated_grid.shape[0]//2, :, :] - real_grid[generated_grid.shape[0]//2, :, :], cmap='bwr', vmin=-200, vmax=200)
                ax[2].set_title("Generated-real")
                # save the figure:
                plt.savefig(PATH_KERNELS+"Kernel_"+str(epoch)+".png")
                plt.close()

                print('fake_path_test:\n', fake_path_test[0])
                print('num_inter: ', num_inter[0].cpu().detach().numpy(), 'energy: ', energy[0].cpu().detach().numpy(), 'masks: ', masks[0].cpu().detach().numpy())

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                for _idx, selected_event in enumerate(fake_path_test[:100]):
                    indices = torch.nonzero(num_inter[_idx]).squeeze()
                    limit_selection = num_inter[_idx].cpu().detach().numpy()
                    selected_event = selected_event.cpu().detach().numpy()
                    nonzero_a = selected_event[:limit_selection]
                    x = nonzero_a[:, 1]
                    y = nonzero_a[:, 2]
                    z = nonzero_a[:, 3]
                    ax.plot(x, y, z, color='red')
                    if _idx == 100:
                        break

                for _idx, selected_event in enumerate(real_path[:100]):
                    limit_selection = num_inter[_idx].cpu().detach().numpy()
                    indices = torch.nonzero(num_inter[_idx]).squeeze()
                    selected_event = selected_event.cpu().detach().numpy()
                    nonzero_a = selected_event[:limit_selection]
                    x = nonzero_a[:, 1]
                    y = nonzero_a[:, 2]
                    z = nonzero_a[:, 3]
                    ax.plot(x, y, z, color='green')
                    if _idx == 100:
                        break

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()

                plt.savefig(PATH_IMAGES+"example_"+str(epoch)+".png")
                plt.close()
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D)
            torch.save(generator.state_dict(), PATH_WEIGHTS+'generator'+MATERIAL+'_epch_'+str(epoch)+'.pth')


# Function to save a checkpoint
def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D):
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict':  discriminator.state_dict(),

        'optimizer_G_state_dict':  optimizer_G.state_dict(),
        'optimizer_D_state_dict':  optimizer_D.state_dict(),

    }
    torch.save(state, PATH_WEIGHTS+"Training_TransfGAN_"+MATERIAL+".pth")

# Function to load a checkpoint
def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D,):
    checkpoint = torch.load(PATH_WEIGHTS+"Training_TransfGAN_"+MATERIAL+".pth")
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    return checkpoint['epoch']


if __name__ == "__main__":
    main()
