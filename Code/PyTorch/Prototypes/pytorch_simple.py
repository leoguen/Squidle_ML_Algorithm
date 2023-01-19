"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import time
import pickle
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets
from torchvision import transforms
#from generate_torch_dataset import kelp_dataset_generator
#from generate_torch_dataset_https import kelp_dataset_generator_https
from datetime import datetime


# Month abbreviation, day and year	
DATE_AND_TIME = datetime.now().strftime("%b-%d-%Y-%H-%M")

transform = transforms.ToTensor()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
BATCHSIZE = 32
CLASSES = 2
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 300
N_VALID_EXAMPLES = BATCHSIZE * 100
print(os.cpu_count())
NUM_WORKERS = 2
SPLIT = [0.8, 0.2]
# Month abbreviation, day and year	
#DATE_AND_TIME = datetime.today().strftime("%b-%d-%Y")


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 24 * 24 * 3 #image size 24*24 pixel and 3 channel (RGB)
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def get_kelp_dataset():
    
    # Create datasets for training & validation, download if necessary
    full_set = torchvision.datasets.ImageFolder(HERE + '/24_images/', transform=transform)

    training_set, validation_set = torch.utils.data.random_split(full_set,SPLIT, generator=torch.Generator().manual_seed(42))
    # Create data loaders for our datasets; shuffle for training and for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    print('Number of training images: {}'.format(len(training_loader.dataset)))
    
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    print('Number of validation images: {}'.format(len(validation_loader.dataset)))
    '''
    kelp_dataset = kelp_dataset_generator(
                        csv_file=os.path.join(ROOT_DIR,"Pytorch_Ecklonia.csv"), 
                        root_dir=os.path.join(ROOT_DIR,"Ecklonia_dataset"),
                        transform=transforms.ToTensor()) 
    train_set, test_set = torch.utils.data.random_split(kelp_dataset,[1200, 299])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCHSIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCHSIZE, shuffle=True)
    '''

    return training_loader, validation_loader

def objective(trial):

    # Generate the model.
    model = define_model(trial)
    model.to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the kelp dataset.
    train_loader, valid_loader = get_kelp_dataset()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data = data.view(data.size(0), -1)
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                #N_VALID_EXAMPLES = len(valid_loader.dataset)
                data = data.view(data.size(0), -1)
                data = data.to(DEVICE)
                target = target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Save a trained model to a file.
        #torch.save(model.state_dict(), "Optuna/optuna_simple_prototype/Trials/{}_trial.pickle".format(trial.number))
        
        #with open(HERE + "/Trials/{}_trial.pickle".format(trial.number), "wb") as fout:
        #    pickle.dump(model, fout)
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(HERE + "/Trials/{}_trial.pth".format(trial.number)) # Save
    return accuracy

def only_keep_best_trial(best_trial_number, trial_value):
    directory = HERE + '/Trials/'
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pth"): 
            if filename != best_trial_number + '_trial.pth':
                os.remove(HERE + '/Trials/' + filename)
                print("Deleted: " + filename)
            else: 
                print("Kept and Renamed: " + filename)
                os.rename(HERE + '/Trials/' + filename, HERE +'/Trials/Saved/best_trial_' + DATE_AND_TIME +"_"+ 
                str(int(trial_value*100))+".pth")



if __name__ == "__main__":
    starttime = time.time()
    print('Start training with: {} and {}'.format(NUM_WORKERS, DEVICE))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: " + str(study.best_trial.number))
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    totaltime = round((time.time() - starttime), 2)
    print('Time {} s, with: {} and {}'.format(totaltime, NUM_WORKERS, DEVICE))
    
    only_keep_best_trial(str(study.best_trial.number), trial.value)


