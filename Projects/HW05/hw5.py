import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# Solving for Q1:
def plot_data(filename):
    data = pd.read_csv(filename)
    
    # Extract the year and number of frozen days
    years = data.iloc[:, 0]
    frozen_days = data.iloc[:, 1]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(years, frozen_days, marker='o', linestyle='-', color='b')

    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    
    plt.savefig("data_plot.jpg")





# Solving for Q2:
def normalize_and_augment(filename):
    data = pd.read_csv(filename)

    years = data.iloc[:, 0].values  
    frozen_days = data.iloc[:, 1]

    m = np.min(years)
    M = np.max(years)
    normalized_years = (years - m) / (M - m)
    
    # Augment the normalized data with a column of 1s
    X_normalized = np.column_stack((normalized_years, np.ones(len(normalized_years))))

    return X_normalized, frozen_days
    



# Solving for Q3:
def closed_form_solution(filename):
    X_normalized, Y = normalize_and_augment(filename)
    
    X_transpose = X_normalized.T
    X_T_X = np.dot(X_transpose, X_normalized)
    X_T_X_inv = np.linalg.inv(X_T_X)
    X_T_Y = np.dot(X_transpose, Y)
    weights = np.dot(X_T_X_inv, X_T_Y)

    return weights
    




# Solving for Q4a:
def gradient_descent_q4a(X_normalized, Y, learning_rate, iterations):
    n = len(Y)
    # Initialize weights to zeros
    weights = np.zeros(2)
    
    # Store weights for every 10 iterations
    all_weights = [weights.copy()]  
    
    for t in range(1, iterations):

        y_prediction = np.dot(X_normalized, weights)
        
        error = y_prediction - Y
        gradient = (1/n) * np.dot(X_normalized.T, error)
        weights = weights - learning_rate * gradient
        
        # Store weights every 10 iterations
        if t % 10 == 0:
            all_weights.append(weights.copy())
    
    return all_weights




# Solving for Q4e:
def gradient_descent_q4e(X_normalized, Y, learning_rate, iterations):
    n = len(Y)
    weights = np.zeros(2)
    
    # Store loss values for plotting
    loss_values = []

    
    for t in range(1, iterations):

        y_prediction = np.dot(X_normalized, weights)
        
        error = y_prediction - Y
        gradient = (1/n) * np.dot(X_normalized.T, error)
        weights = weights - learning_rate * gradient
        
        loss = (1/(2*n)) * np.sum(error ** 2)
        loss_values.append(loss)
    

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations), loss_values, color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss_plot.jpg")




# Solving for Q5:
def predict_2023(filename, weights):
    data = pd.read_csv(filename)
    
    years = data.iloc[:, 0].values  
    m = np.min(years)
    M = np.max(years)
    
    x = 2023  # Year to predict
    x_normalized = (x - m) / (M - m)
    
    w = weights[0]  # Weight
    b = weights[1]  # Bias
    
    # Compute the prediction
    y_hat = w * x_normalized + b

    return y_hat





# Solving for Q6:
def weight_signal(weight):


    if weight > 0:
        symbol = ">"
        interpretation = "Positive correlation between years and ice, indicate colder winters over time"

    elif weight < 0:
        symbol = "<"
        interpretation = "Negative correlation between years and ice, indicate warmer winters over time"

    else:
        symbol = "="
        interpretation = "No correlation found, most likely there is constant ice with relation to the years"

    return symbol, interpretation





# Solving for Q7:
def predict_no_ice(filename, weights):
    data = pd.read_csv(filename)

    w = weights[0]  # Weight
    b = weights[1]  # Bias
    
    years = data.iloc[:, 0].values  
    m = np.min(years)
    M = np.max(years)
    
    x_star = ((-b/w) * (M - m)) + m
    return x_star




if __name__ == "__main__":

    # General Setting:
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])


    # For Q1:
    plot_data(filename)

    
    # For Q2:
    X_normalized, frozen_days = normalize_and_augment(filename)
    print("Q2:")
    print(X_normalized)


    # For Q3:
    weights = closed_form_solution(filename)
    print("Q3:")
    print(weights)


    # For Q4:
    all_weights = gradient_descent_q4a(X_normalized, frozen_days, learning_rate, iterations)
    print("Q4a:")
    for w in all_weights:
        print(w)

    # run gradient descent, every 10 iterations do print(weights)
    print("Q4b: 0.4")
    print("Q4c: 350")
    print("Q4d: Started with a small learning rate 0.05 and increased it gradually while monitoring the convergence of weights. Found that a learning rate of 0.4 allowed gradient descent to converge within 350 iterations (want a generally quick convergence process and since it limits within 500), with the final weights matching the closed-form solution from Question 3 within 0.01. Experimented with larger learning rates but observed divergence.")

    gradient_descent_q4e(X_normalized, frozen_days, learning_rate, iterations)


    # For Q5:
    y_hat = predict_2023(filename, weights)
    print("Q5: " + str(y_hat))


    # For Q6:
    symbol, interpretation = weight_signal(weights[0])
    print("Q6a: " + symbol)
    print("Q6b: " + interpretation)


    # For Q7:
    x_star = predict_no_ice(filename, weights)
    print("Q7a: " + str(x_star))
    print("Q7b: " + "x_star is not a compelling one since it is not that accurate. The limitations are primarily assume linear relationship between years and ice days, which is not comprehensive. It is also needed to add additional elements to take into account such as the co2 emission and more historical dataset")

