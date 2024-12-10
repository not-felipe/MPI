from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix dimensions
N = 1000

# Matrix A and B
A = None
B = np.random.rand(N, N)

start_time_total = MPI.Wtime()

# Root process initializes matrix A
if rank == 0:
    A = np.random.rand(N, N)
else:
    A = np.empty((N, N), dtype=np.float64)

# Broadcast the entire matrix A
start_time_bcast_A = MPI.Wtime()
comm.Bcast(A, root=0)
end_time_bcast_A = MPI.Wtime()

# Divide rows among processes
rows_per_process = N // size
start_idx = rank * rows_per_process
end_idx = start_idx + rows_per_process

A_local = A[start_idx:end_idx, :]

# Perform the local computation
start_time_calc = MPI.Wtime()
C_local = np.dot(A_local, B)
end_time_calc = MPI.Wtime()

# Gather results
start_time_gather = MPI.Wtime()
C = None
if rank == 0:
    C = np.empty((N, N), dtype=np.float64)
comm.Gather(C_local, C, root=0)
end_time_gather = MPI.Wtime()

end_time_total = MPI.Wtime()

if rank == 0:
    print(f"- Broadcast time (Bcast A): {end_time_bcast_A - start_time_bcast_A:.4f} seconds")
    print(f"- Computation time: {end_time_calc - start_time_calc:.4f} seconds")
    print(f"- Gather time: {end_time_gather - start_time_gather:.4f} seconds")
    print(f"- Total time: {end_time_total - start_time_total:.4f} seconds")
