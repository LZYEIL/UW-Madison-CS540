import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if(training):
        train_set=datasets.FashionMNIST("./data",train=True,
            download=True,transform=custom_transform) 
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    if(training == False):
        test_set=datasets.FashionMNIST("./data", train=False,
            transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)
    
    return loader




def build_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
    )
    return model



def build_deeper_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):  # loop over the dataset T times
        correct = 0
        total_loss = 0.0
        total_samples = 0
        
        for data, labels in train_loader:
            # zero the parameter gradients
            opt.zero_grad()
            
            # forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # backward pass and optimize
            loss.backward()
            opt.step()
            
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)  # Get actual batch size
            total_samples += batch_size
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item() * batch_size
        
        # Calculate average loss per sample
        avg_loss = total_loss / total_samples
        
        # Print epoch statistics with required format
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total_samples}({correct/total_samples*100:.2f}%) Loss: {avg_loss:.3f}")


      




def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    model.eval()  
    
    correct = 0
    total_loss = 0.0
    total_samples = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data, labels in test_loader:
            # Forward pass only
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # In here, the total samples are divided into small
            # batches and each batch has different number of samples in it
            #That's why we need to compute the total_loss here for each batch and
            #add them all at the end to divide by the total sample size
            #same goes to the logic of accuracy
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100

    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


    


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    model.eval()
    
    with torch.no_grad():
        # Extract the image at the specified index
        image = test_images[index].unsqueeze(0)  # Add batch dimension
        
        logits = model(image)
        
        probabilities = F.softmax(logits, dim=1)
        
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        # Convert probabilities to percentages
        top3_prob = top3_prob.squeeze().numpy() * 100
        top3_indices = top3_indices.squeeze().numpy()
        
        for i in range(3):
            label = class_names[top3_indices[i]]
            prob = top3_prob[i]
            print(f"{label}: {prob:.2f}%")



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

    # From function 1:
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)

    model = build_deeper_model()

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = True)

    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 1)
    
