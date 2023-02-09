import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch.utils.data.dataset import Dataset
import glob
from PIL import Image
import cv2
from torchvision import transforms

IMG_PATH = '/pvol/Ecklonia_Database/'
IMG_SIZE = 24

class GeneralDataset(Dataset):
    def __init__(self, img_size, test_list, test, inception):
        self.inception = inception
        if test: # True test dataset is returned
            # Add unpadded and padded entries to data
            self.data = test_list['test','test']
            #self.data = self.data + test_list[1]
            self.class_map = {"Ecklonia" : 0, "Others": 1}
        else: 
            self.imgs_path = IMG_PATH + str(img_size)+ '_images/'
            file_list = [self.imgs_path + 'Others', self.imgs_path + 'Ecklonia']
            self.data = []
            for class_path in file_list:
                class_name = class_path.split("/")[-1]
                for img_path in glob.glob(class_path + "/*.jpg"):
                    self.data.append([img_path, class_name])
            
            # Delete all entries that are used in test_list
            #del_counter = len(self.data)
            #print('Loaded created dataset of {} entries. \nNow deleting {} duplicate entries.'.format(len(self.data), len(test_list[0])))
            #for test_entry in (test_list[0]):
            #    if test_entry in self.data:
            #        self.data.remove(test_entry)

            #print('Deleted {} duplicate entries'.format(del_counter-len(self.data)))
            self.class_map = {"Ecklonia" : 0, "Others": 1}
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = self.class_map[class_name]
        #train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #!InceptionV3
        img = Image.fromarray(img)
        if self.inception:
            train_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transforms = transforms.Compose([
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        img_tensor = train_transforms(img)
        #print(type(img_tensor))
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class Net(nn.Module):
    """CNN for the MNIST dataset of handwritten digits.
    Attributes:
        - convs (torch.nn.modules.container.ModuleList):   List with the convolutional layers
        - conv2_drop (torch.nn.modules.dropout.Dropout2d): Dropout for conv layer 2
        - out_feature (int):                               Size of flattened features
        - fc1 (torch.nn.modules.linear.Linear):            Fully Connected layer 1
        - fc2 (torch.nn.modules.linear.Linear):            Fully Connected layer 2
        - p1 (float):                                      Dropout ratio for FC1
    Methods:
        - forward(x): Does forward propagation
    """
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1):
        """Parameters:
            - trial (optuna.trial._trial.Trial): Optuna trial
            - num_conv_layers (int):             Number of convolutional layers
            - num_filters (list):                Number of filters of conv layers
            - num_neurons (int):                 Number of neurons of FC layers
            - drop_conv2 (float):                Dropout ratio for conv layer 2
            - drop_fc1 (float):                  Dropout ratio for FC1
        """
        super(Net, self).__init__() # Initialize parent class
        in_size = IMG_SIZE # Input image size 
        kernel_size = 3 # Convolution filter size

        # Define the convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(3, num_filters[0], kernel_size=(3, 3))])  # List with the Conv layers
        out_size = in_size - kernel_size + 1 # Size of the output kernel
        out_size = int(out_size / 2) # Size after pooling
        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=(3, 3)))
            out_size = out_size - kernel_size + 1 # Size of the output kernel
            out_size = int(out_size/2) # Size after pooling

        self.conv2_drop = nn.Dropout2d(p=drop_conv2) # Dropout for conv2
        self.out_feature = num_filters[num_conv_layers-1] * out_size * out_size # Size of flattened features
        self.fc1 = nn.Linear(self.out_feature, num_neurons) # Fully Connected layer 1
        self.fc2 = nn.Linear(num_neurons, 2) # Fully Connected layer 2
        self.p1 = drop_fc1 # Dropout ratio for FC1

        # Initialize weights with the the initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        """Forward propagation.
        Parameters:
            - x (torch.Tensor): Input tensor of size [N,1,28,28]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,10]
        """
        for i, conv_i in enumerate(self.convs):  # For each convolutional layer
            if i == 2:  # Add dropout if layer 2
                x = F.relu(F.max_pool2d(self.conv2_drop(conv_i(x)), 2))  # Conv_i, dropout, max-pooling, RelU
            else:
                x = F.relu(F.max_pool2d(conv_i(x), 2))                   # Conv_i, max-pooling, RelU

        x = x.view(-1, self.out_feature) # Flatten tensor
        x = F.relu(self.fc1(x)) # FC1, RelU
        x = F.dropout(x, p=self.p1, training=self.training)  # Apply dropout after FC1 only when training
        x = self.fc2(x) # FC2

        return F.log_softmax(x, dim=1) # log(softmax(x))


def train(network, optimizer):
    """Trains the model.
    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    for batch_i, (data, target) in enumerate(train_loader):  # For each batch

        optimizer.zero_grad() # Clear gradients
        output = network(data.to(device)) # Forward propagation
        loss = F.nll_loss(output, target.to(device)) # Compute loss (negative log likelihood: −log(y))
        loss.backward() # Compute gradients
        optimizer.step() # Update weights


def valid(network):
    """valids the model.
    Parameters:
        - network (__main__.Net): The CNN
    Returns:
        - accuracy_valid (torch.Tensor): The valid accuracy
    """
    network.eval() # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    valid_len = len(val_loader.dataset)
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, (data, target) in enumerate(val_loader):  # For each batch
            output = network(data.to(device)) # Forward propagation
            pred = output.data.max(1, keepdim=True)[1] # Find max value in each row, return indexes of max values
            #! Target has the correct values and pred the predicted values. Could prob use for loop and get TP, TN, FP, FN
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()  # Compute correct predictions

    accuracy_valid = correct / valid_len

    return accuracy_valid

def objective(trial):
    """Objective function to be optimized by Optuna.
    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.
    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The valid accuracy. Parameter to be maximized.
    """

    # Define range of values to be valided for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 3)  # Number of convolutional layers
    num_filters = [int(trial.suggest_int("num_filter_"+str(i), 16, 128, 16)) for i in range(num_conv_layers)] # Number of filters for the convolutional layers
    num_neurons = trial.suggest_int("num_neurons", 10, 400, 10)  # Number of neurons of FC1 layer
    drop_conv2 = trial.suggest_float("drop_conv2", 0.2, 0.5) # Dropout for convolutional layer 2
    drop_fc1 = trial.suggest_float("drop_fc1", 0.2, 0.5) # Dropout for FC1 layer

    # Generate the model
    model = Net(trial, num_conv_layers, num_filters, num_neurons, drop_conv2,  drop_fc1).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-8, 1e-4, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer)  # Train the model
        accuracy = valid(model)   # Evaluate the model
        print(accuracy)
        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save('/pvol/MLP_Trials/{}_trial.pth'.format(trial.number)) # Save

    return accuracy


if __name__ == '__main__':

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 10          # Number of training epochs
    batch_size = 64  # Batch size for training data
    batch_size_valid = 64 # Batch size for validing data
    number_of_trials = 100 # Number of Optuna trials
    limit_obs = True       # Limit number of observations for faster computation
    num_workers=os.cpu_count()        # Define number of CPUs working
    split = [0.8, 0.2]     # Dataset split Training/Validation

    # *** Note: For more accurate results, do not limit the observations.
    #           If not limited, however, it might take a very long time to run.
    #           Another option is to limit the number of epochs. ***
    

    # Make runs repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)


    HERE = os.path.dirname(os.path.abspath(__file__))
    test_list = [0]
    train_val_set = GeneralDataset(img_size = IMG_SIZE, test_list = test_list, test = False, inception = False)
    
    training_set, validation_set, throw_away = torch.utils.data.random_split(train_val_set,[0.090, 0.010, 0.9], generator=torch.Generator().manual_seed(123))

    
    # Create data loaders for our datasets; shuffle for training and for validation
    train_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle=True, num_workers=os.cpu_count())
    
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size, shuffle=False, num_workers=os.cpu_count())


    print('Number of training images: {}'.format(len(train_loader.dataset)))
    
    print('Number of validation images: {}'.format(len(val_loader.dataset)))

    if limit_obs:  # Limit number of observations
        number_of_train_examples =len(train_loader.dataset)/100  # Max train observations
        number_of_valid_examples = len(val_loader.dataset)/100 # Max valid observations
    else:
        number_of_train_examples = len(train_loader.dataset) # Max train observations
        number_of_valid_examples = len(val_loader.dataset) # Max valid observations

    # Create an Optuna study to maximize valid accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    #-------------------------------------------------------------------------
    # Results
    #-------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    #df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
