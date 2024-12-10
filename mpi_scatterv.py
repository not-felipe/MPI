from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix dimensions
N = 1000  # Size of the square matrix

# Matrix A and B on the root process
A = None
B = np.random.rand(N, N)

# Initialize matrix A only on the root process
start_time_total = MPI.Wtime()

if rank == 0:
    A = np.random.rand(N, N)

# Scatter the rows of matrix A using Scatterv
start_time_scatterv = MPI.Wtime()
rows_per_process = [N // size + (1 if i < N % size else 0) for i in range(size)]
displs = [sum(rows_per_process[:i]) * N for i in range(size)]  # Calculate displacements
sendcounts = [rows * N for rows in rows_per_process]  # Flattened row counts
A_local = np.empty((rows_per_process[rank], N), dtype=np.float64)

if rank == 0:
    A_flat = A.flatten()
else:
    A_flat = None

comm.Scatterv([A_flat, sendcounts, displs, MPI.DOUBLE], A_local, root=0)
end_time_scatterv = MPI.Wtime()

# Broadcast matrix B to all processes
start_time_bcast = MPI.Wtime()
comm.Bcast(B, root=0)
end_time_bcast = MPI.Wtime()

# Perform the local matrix multiplication
start_time_calc = MPI.Wtime()
C_local = np.dot(A_local, B)
end_time_calc = MPI.Wtime()

# Gather the result matrix C back to the root process using Gatherv
start_time_gatherv = MPI.Wtime()
C = None
recvcounts = [rows * N for rows in rows_per_process]
displs = [sum(rows_per_process[:i]) * N for i in range(size)]

if rank == 0:
    C = np.empty((N, N), dtype=np.float64)

comm.Gatherv(C_local, [C, recvcounts, displs, MPI.DOUBLE], root=0)
end_time_gatherv = MPI.Wtime()

# Print the result if root
end_time_total = MPI.Wtime()

if rank == 0:
    print("Matrix multiplication completed.")
    print("Top-left 10x10 section of the result matrix:")
    print(C[:10, :10])  # Print the top-left 10x10 section of the result matrix
    print(f"\nTimings:")
    print(f" - Distribution time (Scatterv): {end_time_scatterv - start_time_scatterv:.4f} seconds")
    print(f" - Broadcast time (Bcast): {end_time_bcast - start_time_bcast:.4f} seconds")
    print(f" - Computation time: {end_time_calc - start_time_calc:.4f} seconds")
    print(f" - Gather time (Gatherv): {end_time_gatherv - start_time_gatherv:.4f} seconds")
    print(f" - Total time: {end_time_total - start_time_total:.4f} seconds")
