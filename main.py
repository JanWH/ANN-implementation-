import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read training data from Excel file
training_dataframe = pd.read_excel('normalized_data.xlsx')
training_data = training_dataframe.values.tolist()

# Define network parameters
No_hidden_nodes = 6
No_inputs = 8
learning_param = 0.1

# Ask user for random initial weights or not
first_run = input("Do you want random initial weights? (y/n)")
if first_run == "y":
    # Randomly initialize weights for hidden layer and output layer
    nodes = (np.random.uniform((-2/No_inputs),(2/No_inputs),[No_hidden_nodes, No_inputs+1])).tolist()
    outnode = (np.random.uniform((-2/No_hidden_nodes),(2/No_hidden_nodes),[1,No_hidden_nodes+1])).tolist()
else:
    nodes = []  # Placeholder for custom weights
    outnode = []  # Placeholder for custom weights

# Ask user for number of epochs
epochs = int(input("How many epochs to run?:"))

# Sigmoid activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Squared error function
def SE(result, real_val):
    return (result-real_val)**2

# Forward propagation
def forward(nodes, inputs):
    weighted_out = []
    for node in nodes:
        total = 0
        for i in range(len(node)-1):
            total += node[i]*inputs[i]
        total += node[len(node)-1]  # Add bias weight
        weighted_out.append(total)
    return [sigmoid(weight) for weight in weighted_out]

# Calculate delta for output layer
def deltaOut(result, real_val):
    return (real_val-result)*(result*(1-result))

# Calculate delta for hidden layer
def deltaHidden(weight, deltaOut, nodeOut):
    return (weight*deltaOut*(nodeOut*(1-nodeOut)))

# Backward propagation for updating weights
def backward(result, real_val, learning_param, outnode, hidden_results, nodes, inputs):
    no_hidden_inputs = len(outnode[0])
    no_inputs = len(nodes[0])
    delta = deltaOut(result, real_val)
    for i in range(len(nodes)):
        hidden_delta = deltaHidden(outnode[0][i], delta, hidden_results[i])
        for j in range(no_inputs-1):
            nodes[i][j] -= learning_param*hidden_delta*inputs[j]
        nodes[i][no_inputs-1] -= learning_param*hidden_delta  # Update bias
    for i in range(no_hidden_inputs-1):
        outnode[0][i] +=  learning_param*delta*hidden_results[i]
    outnode[0][no_hidden_inputs-1] += learning_param*delta  # Update bias
    return outnode, nodes

# Lists to store results of forward pass
hidden_results = []
out_results = []

# Training loop
j = 0
for i in range(epochs):
    real_val = training_data[j][No_inputs]
    hidden_results.append(forward(nodes, training_data[j]))
    out_results.append(forward(outnode, hidden_results[i]))
    outnode, nodes = backward(out_results[i][0], real_val, learning_param, outnode, hidden_results[i], nodes, training_data[j])
    j += 1
    if j > (len(training_data)-1):
        j = 0

# Save results to Excel file
resultsdf = pd.DataFrame(data = out_results)
resultsdf.to_excel('results.xlsx')
