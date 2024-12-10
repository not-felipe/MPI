from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix dimensions
N = 1000  # Size of the square matrix
rows_per_process = N // size

# Matrix A and B
A = None
B = np.random.rand(N, N)

start_time_total = MPI.Wtime()

if rank == 0:
    A = np.random.rand(N, N)

# Scatter rows of matrix A
start_time_scatter = MPI.Wtime()
A_local = np.empty((rows_per_process, N), dtype=np.float64)
comm.Scatter(A, A_local, root=0)
end_time_scatter = MPI.Wtime()

# Broadcast matrix B to all processes
start_time_bcast = MPI.Wtime()
comm.Bcast(B, root=0)
end_time_bcast = MPI.Wtime()

# Perform the local matrix multiplication
start_time_calc = MPI.Wtime()
C_local = np.dot(A_local, B)
end_time_calc = MPI.Wtime()

# Use Reduce to sum the local results into the root process
start_time_reduce = MPI.Wtime()
C = np.zeros_like(C_local)  # Create a buffer for the reduction
comm.Reduce(C_local, C, op=MPI.SUM, root=0)
end_time_reduce = MPI.Wtime()

end_time_total = MPI.Wtime()

# Print results and timings
if rank == 0:
    print("Matrix multiplication completed.")
    print("Top-left 10x10 section of the result matrix:")
    print(C[:10, :10])  # Print the top-left 10x10 section of the result matrix
    print(f"\nTimings:")
    print(f" - Scatter time: {end_time_scatter - start_time_scatter:.4f} seconds")
    print(f" - Broadcast time (Bcast): {end_time_bcast - start_time_bcast:.4f} seconds")
    print(f" - Computation time: {end_time_calc - start_time_calc:.4f} seconds")
    print(f" - Reduce time: {end_time_reduce - start_time_reduce:.4f} seconds")
    print(f" - Total time: {end_time_total - start_time_total:.4f} seconds")
