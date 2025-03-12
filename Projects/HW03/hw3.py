from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x_Matrix = np.load(filename)  #Here x_Matrix represents the matrix(dataset)
    return x_Matrix - np.mean(x_Matrix, axis=0)
    


def get_covariance(dataset):
    n = dataset.shape[0]  #Get the value n
    return (1 / (n - 1)) * np.dot(np.transpose(dataset), dataset)


def get_eig(S, k):
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[S.shape[0] - k, S.shape[0] - 1])
    eigenvalues = eigenvalues[::-1]  #Reverse the eigenvalues
    eigenvectors = eigenvectors[:, ::-1] #Reverse the eigenvectors (This one is a matrix)
    Lambda = np.diag(eigenvalues)
    return Lambda, eigenvectors


def get_eig_prop(S, prop):
    eigenvalues, eigenvectors = eigh(S)
    
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    total = np.sum(eigenvalues)
    formula = eigenvalues / total

    m = 0  
    for variance in formula:
        if variance > prop:  
            m += 1  

    return get_eig(S, m)        





def project_and_reconstruct_image(image, U):
    projection = np.dot(np.transpose(U), image)  
    reconstruct = np.dot(U, projection)  
    
    return reconstruct



def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    #Reshape all of them
    im_orig_fullres = im_orig_fullres.reshape((218,178,3))
    im_orig = im_orig.reshape(60, 50)
    im_reconstructed = im_reconstructed.reshape(60, 50)

    plt.sca(ax1)  
    plt.imshow(im_orig_fullres, aspect='equal')
    ax1.set_title("Original High Res")

    plt.sca(ax2)  
    im2 = plt.imshow(im_orig, aspect='equal', cmap='gray')
    ax2.set_title("Original")
    plt.colorbar(im2, ax=ax2, cmap='gray')

    plt.sca(ax3)  
    im3 = plt.imshow(im_reconstructed, aspect='equal', cmap='gray')
    ax3.set_title("Reconstructed")
    plt.colorbar(im3, ax=ax3, cmap='gray')

    return fig, ax1, ax2, ax3



def perturb_image(image, U, sigma):
    projection = np.dot(np.transpose(U), image)  
    
    perturbation = np.random.normal(0, sigma, projection.shape)  
    pertur = projection + perturbation  
    perturbed_image = np.dot(U, pertur)  
    return perturbed_image  





#The below is the main method:
# if __name__ == "__main__":
#     datadd = load_and_center_dataset('celeba_60x50.npy')
#     covariance_Matrix = get_covariance(datadd)
#     Lambda, U = get_eig(covariance_Matrix, 50)


#     x = datadd[34]
#     x_fullres = np.load('celeba_218x178x3.npy')[34]
#     reconstructed = project_and_reconstruct_image(x, U)
#     fig, ax1, ax2, ax3 = display_image(x_fullres, x, reconstructed)

#     x_perturbed = perturb_image(x, U, sigma=1000)
#     fig, ax1, ax2, ax3 = display_image(x_fullres, x, x_perturbed)



#     plt.show()


    






  

