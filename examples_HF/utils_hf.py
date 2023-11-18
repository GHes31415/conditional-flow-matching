import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import os
import imageio

def gif_quiver(model,save_dir,xx0,num_samples,N = 50):

    # Generate a grid of points

    nx = 25
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, nx)
    X, Y = np.meshgrid(x, y)
    N = 50

    xx = xx0.clone()
    tt = torch.ones((num_samples,1))/N*0

    for i in range(N+1):
        t = torch.ones((nx*nx,1))/N*i
        # Convert the grid to PyTorch tensors
        X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        Y_torch = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

        X_0 = torch.cat([X_torch, Y_torch], dim=-1).reshape((nx*nx, 2))

        # Concatenate the tensors and pass them through the model
        inputs = torch.cat([X_0,t],dim = -1)
        outputs = model(inputs).reshape((nx, nx, 2))

        # Convert the outputs to numpy arrays
        U = outputs[:,:, 0].detach().numpy()
        V = outputs[:,:, 1].detach().numpy()

        # Compute the magnitude of the vectors
        magnitude = np.sqrt(U**2 + V**2)

        plt.scatter(xx[:,0],xx[:,1],s = 4)

        vector_field = model(torch.cat([xx,tt],dim = -1)).detach().numpy()

        xx = xx + vector_field/N

        tt = tt + torch.ones((num_samples,1))/N

        # plt.scatter(xx[:,0],xx[:,1],s = 4)

        # plt.scatter(gg[:,0],gg[:,1],s = 4)

        # Create a quiver plot
        plt.quiver(X, Y, U, V, magnitude, cmap='viridis')
        plt.colorbar(label='Magnitude')
        # plt.show()
        plt.savefig(os.path.join(save_dir,f"quiver_MLP_25_000_{i}.png"))
        plt.close()
    images = [imageio.imread(os.path.join(save_dir,f"quiver_MLP_25_000_{i}.png")) for i in range(N)]
    imageio.mimsave(os.path.join(save_dir,f"quiver_MLP_25_000.gif"), images,duration = 5)

def particle_gif(model,savedir,xx,gg,N = 50):
    dt = 1/N
    j = 0
    for i in np.linspace(0,1,N):
        
        t = i*torch.ones((1000,1))
        
        
        # Concatenate the tensors and pass them through the model
        inputs = torch.cat([xx,t],dim = -1)
        outputs = model(inputs)

        plt.scatter(xx[:,0].detach(),xx[:,1].detach(),s = 4)
        plt.scatter(gg[:,0],gg[:,1],s = 4)

        xx = xx+outputs*dt
        
        plt.savefig(os.path.join(savedir,f"figs_gif/parts_MLP_15_000_{j}.png"))
        j+=1
        plt.close()
        # plt.show()
    images = [imageio.imread(os.path.join(savedir,f"figs_gif/parts_MLP_15_000_{i}.png")) for i in range(N)]
    imageio.mimsave(os.path.join(savedir,f"figs_gif/point_evol_MLP_15_000.gif"), images, duration = 5)