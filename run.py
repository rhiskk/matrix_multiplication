import time
import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import psutil

np.set_printoptions(threshold=10**6)

# Measure the memory and CPU usage over time
def performance_measurement(queue, stop_event):
    memory = []
    cpu = []
    time_stamps = []
    start_time = time.time()

    while not stop_event.is_set():
        memory.append(psutil.virtual_memory()[3] / 1024**3)
        cpu.append(psutil.cpu_percent(interval=1))
        time_stamps.append(time.time()-start_time)
        
    queue.put((time_stamps, memory, cpu))

# Generate matrices
def generate_matrices():
    A = np.random.rand(10**6, 10**3)
    B = np.random.rand(10**3, 10**6)
    C = np.random.rand(10**6, 1)
    return (A, B, C)

# Perform matrix multiplication
def matrix_multiplication(A, B, C):
    BC = np.dot(B, C)
    D = np.dot(A, BC)
    return D

# Plot the memory and CPU usage over time
def plot_performance(path, time_stamps, memory, cpu):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(time_stamps, memory,'tab:orange')
    ax1.set_ylabel('Memory Usage (GB)')
    ax1.set_title('Memory Usage Over Time')
    ax2.plot(time_stamps, cpu, 'tab:green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.set_title('CPU Usage Over Time')
    plt.tight_layout()
    plt.savefig(path + 'performance.png')
    plt.clf()

# Plot the CDF of A
def plot_cdf(path, A):
    hist, bin_edges = np.histogram(A, bins=1000, density=True)
    cdf = np.cumsum(hist)
    y = cdf / cdf[-1]
    x = bin_edges[1:]
    plt.plot(x, y, 'tab:blue')
    plt.xlabel('Value in A less than x')
    plt.ylabel('Probability')
    plt.title('CDF of A')
    plt.savefig(path + 'cdf.png')
    plt.clf()

def main():
    # Start a process to measure performance
    queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    process = multiprocessing.Process(target=performance_measurement, args=(queue, stop_event))
    process.start()

    A, B, C = generate_matrices()
    D = matrix_multiplication(A, B, C)

    # Stop the process
    stop_event.set()
    process.join()
    time_stamps, memory, cpu = queue.get()

    # Create a directory to save results
    path = 'results/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Save results into a file
    file = open(path + 'd.txt', 'w')
    file.write(str(D))
    file.close()

    plot_performance(path, time_stamps, memory, cpu)
    plot_cdf(path, A)

if __name__ == '__main__':
    main()