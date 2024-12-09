from mpi4py import MPI
import numpy as np

# Inicializa MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configurações
rows, cols = 10000, 5000
data = None

# Início da medição
start_time = MPI.Wtime()

# Geração e distribuição da matriz por meio do Scatterv
if rank == 0:
    data = np.random.rand(rows, cols).astype(np.float64)
    
base_rows = rows // size
extra_rows = rows % size
send_counts = [(base_rows + (1 if i < extra_rows else 0)) * cols for i in range(size)]
displs = [sum(send_counts[:i]) for i in range(size)]
local_rows = base_rows + (1 if rank < extra_rows else 0)
local_data = np.empty((local_rows, cols), dtype=np.float64)
comm.Scatterv([data, send_counts, displs, MPI.DOUBLE], local_data, root=0)

# Fim da distribuição
dist_time = MPI.Wtime()

# Cálculo local
local_sum = np.sum(local_data)
local_sum_sq = np.sum(local_data ** 2)
local_count = local_data.size

# Combinação dos resultados
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
global_sum_sq = comm.reduce(local_sum_sq, op=MPI.SUM, root=0)
global_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Fim do cálculo
calc_time = MPI.Wtime()

# Cálculo global no mestre
if rank == 0:
    global_mean = global_sum / global_count
    global_variance = (global_sum_sq / global_count) - global_mean ** 2
    global_std = np.sqrt(global_variance)
    print(f"Média global: {global_mean}, Desvio padrão global: {global_std}")

# Fim do programa
end_time = MPI.Wtime()

# Relatório de tempos
if rank == 0:
    print(f"Tempo de distribuição: {dist_time - start_time:.4f} s")
    print(f"Tempo de cálculo: {calc_time - dist_time:.4f} s")
    print(f"Tempo total: {end_time - start_time:.4f} s")
