import numpy as np
import matplotlib.pyplot as plt


def trajectory(S0 = 100, mu = 0.05, sigma = 0.2, T = 1.0, dt = 0.01,M = 1000):
    # Parameters
    S0 = 100        
    mu = 0.05       
    sigma = 0.2
    T = 1.0        
    dt = 0.01       
    N = int(T / dt) 
    M = 1000          # number of paths

    # Time vector
    t = np.linspace(0, T, N + 1)

    # Simulate GBM paths
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for i in range(1, N + 1):
        Z = np.random.standard_normal(M)  # standard normal random variables
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)


    print(len(S))
    # Plot
    plt.figure(figsize=(10, 6))
    for j in range(M):
        plt.plot(t, S[j], lw=1)
    plt.title("Geometric Brownian Motion Simulation")
    plt.xlabel("Time (Years)")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.show()
    print("done")
    exit()

trajectory()