# Standard library imports
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# PyTorch related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Local imports from custom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import GeneratorNumIntEnergyDirection2, ViTwMask2
from src.utils import generate_3D_images
from src.dataloader import DataSetNumIntEnergyDirection

LATENT_DIM = 100
NUM_FEATURES = 4  # Energy, X, Y, Z


def parse_args():
    p = argparse.ArgumentParser(description="Train the positron-path GAN for one emitter and one material.")
    p.add_argument("--data-dir", required=True,
                   help="Folder with the GATE phase-space files positrons_<k>.npy, each of shape (events, steps, 4).")
    p.add_argument("--output-dir", default="runs", help="Where checkpoints and figures are written.")
    p.add_argument("--material", default="Water", help="Material label, used in output file names (e.g. Water, RibBone, Lung).")
    p.add_argument("--emitter", default="F18", choices=["F18", "Ga68"],
                   help="Positron emitter. Sets the path length (18 steps for F18, 30 for Ga68) unless --num-steps is given.")
    p.add_argument("--num-steps", type=int, default=None, help="Maximum number of interactions per path.")
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4, help="Generator learning rate (the discriminator uses 3x).")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--min-selection", type=int, default=0,
                   help="Use files positrons_<k>.npy with k > MIN_SELECTION ...")
    p.add_argument("--max-selection", type=int, default=20, help="... and k < MAX_SELECTION.")
    p.add_argument("--device", default=None, help="cuda or cpu. Default: cuda if available.")
    return p.parse_args()


# Define loss functions
def Energy_loss(generated_data):
    below_zero_penalty = torch.relu(-generated_data).sum(dim=-1)
    diff = generated_data[:, :, 1:] - generated_data[:, :, :-1]
    non_decreasing_penalty = torch.relu(diff).sum(dim=-1)
    total_penalty = non_decreasing_penalty + below_zero_penalty
    total_penalty = 0.005 * total_penalty.mean()
    return total_penalty

Criterion_Angle = nn.CosineSimilarity(dim=1, eps=1e-6) 

def train_model_step(batch_size, device, discriminator, generator, d_optimizer, g_optimizer, real_paths, num_inter, energy, masks, start_vectors):
    
    d_optimizer.zero_grad()
   

    real_paths_d = real_paths #torch.cat((output, real_paths), dim=1)
    # Train discriminator
    real_validity = discriminator(real_paths_d, num_inter, energy, masks)
    

    z = Variable(torch.randn(batch_size, LATENT_DIM)).to(device)
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
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    num_steps = args.num_steps or (30 if args.emitter == 'Ga68' else 18)
    material = args.material
    path_base = os.path.join(args.output_dir, args.emitter, material)
    path_weights = os.path.join(path_base, 'Weights', f'TransfGAN_{material}/')
    path_images = os.path.join(path_base, 'Images', f'TransfGAN_{material}/')
    path_kernels = os.path.join(path_base, 'Kernels', f'TransfGAN_{material}/')
    for path in [path_weights, path_images, path_kernels]:
        os.makedirs(path, exist_ok=True)

    # Load data, create dataset and dataloader
    # Load data and create dataset and dataloader
    Input_data_list = [np.load(os.path.join(args.data_dir, f)) for f in sorted(os.listdir(args.data_dir)) if ("positrons_" in f and int(f.replace("positrons_", "").replace(".npy", "")) < args.max_selection) and int(f.replace("positrons_", "").replace(".npy", "")) > args.min_selection]
    dataset     = DataSetNumIntEnergyDirection(Input_data_list, num_steps=num_steps, num_features=NUM_FEATURES)
    dataloader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    tdataloader = DataLoader(dataset, batch_size=10000, shuffle=True)

    # Define the generator and discriminator
    generator      = GeneratorNumIntEnergyDirection2(seq_len=num_steps).to(device)
    discriminator  = ViTwMask2(image_size=num_steps, patch_size=1, num_classes=1, channels=NUM_FEATURES, dim=64, depth=3, heads=4, mlp_dim=128).to(device)

    # Initialize optimizers
    lr = args.lr
    optimizer_G  = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_D  = torch.optim.Adam(discriminator.parameters(), lr=3*lr, betas=(0.9, 0.999))

    # Load checkpoint if exists
    Start_EPOCH = 0
    checkpoint_file = os.path.join(path_weights, "Training_TransfGAN_" + material + ".pth")
    if os.path.isfile(checkpoint_file):
        Start_EPOCH = load_checkpoint(checkpoint_file, generator, discriminator, optimizer_G, optimizer_D, device) + 1
    else:
        print('----------Start from scratch------------')
    # Start training loop
    for epoch in range(Start_EPOCH, args.epochs):
        print('Starting epoch {}...'.format(epoch), end=' ')
        step = 0

        for i, (real_path, num_inter, energy, masks, start_vector) in enumerate(dataloader):
            if i == len(dataloader) - 1 and num_inter.shape[0] < args.batch_size:
                break
            step = epoch * len(dataloader) + i + 1

            real_path = Variable(real_path).type(torch.float32).to(device)
            num_inter = Variable(num_inter).type(torch.long).to(device)
            energy = Variable(energy).type(torch.float32).to(device)
            masks = Variable(masks).type(torch.float32).to(device)
            start_vector = Variable(start_vector).type(torch.float32).to(device)

            generator.train()
            g_loss, d_loss =  train_model_step(len(real_path), device, discriminator, generator, optimizer_D, optimizer_G, real_path, num_inter, energy, masks, start_vector)
            if i % 500 == 0:
                print('scalars', {'g_loss': g_loss, 'd_loss': (d_loss)}, step)
        if epoch % 1 == 0:
            with torch.no_grad():
                for i, batch in enumerate(tdataloader):
                    real_path, num_inter, energy, masks, start_vector = batch
                    break

                num_inter = num_inter.type(torch.long).to(device)#[:100]
                energy = energy.type(torch.float32).to(device)#[:100]
                masks = masks.type(torch.float32).to(device)#[:100]
                real_path = real_path.type(torch.float32).to(device)#[:100]
                start_vector = start_vector.type(torch.float32).to(device)#[:100]

                hidden = Variable(torch.randn(num_inter.shape[0], LATENT_DIM)).to(device)
                fake_path_test = generator(hidden, num_inter, energy, masks, start_vector).to(device)
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
                plt.savefig(path_kernels+"Kernel_"+str(epoch)+".png")
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

                plt.savefig(path_images+"example_"+str(epoch)+".png")
                plt.close()
            save_checkpoint(checkpoint_file, epoch, generator, discriminator, optimizer_G, optimizer_D)
            torch.save(generator.state_dict(), path_weights+'generator'+material+'_epch_'+str(epoch)+'.pth')


# Function to save a checkpoint
def save_checkpoint(checkpoint_file, epoch, generator, discriminator, optimizer_G, optimizer_D):
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict':  discriminator.state_dict(),

        'optimizer_G_state_dict':  optimizer_G.state_dict(),
        'optimizer_D_state_dict':  optimizer_D.state_dict(),

    }
    torch.save(state, checkpoint_file)

# Function to load a checkpoint
def load_checkpoint(checkpoint_file, generator, discriminator, optimizer_G, optimizer_D, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    return checkpoint['epoch']


if __name__ == "__main__":
    main()
