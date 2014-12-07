compile the source on the remote host:

```
ssh compiler
module load impi
mpicxx floyd.cpp -o ~/_scratch/floyd -DNO_OMP
mpicxx floyd.cpp -o ~/_scratch/floyd_omp -fopenmp
exit
```

run the binary:

1) MPI
```
module load impi
sbatch -p test -N 32 --ntasks-per-node=8 impi ~/floyd input-example.txt # -N 32 means 32 tasks
```

2) MPI + OpenMP
```
module load impi
sbatch -p test -N 32 --ntasks-per-node=1 impi ~/floyd_omp input-example.txt 8 # -N 32 means 32 tasks; 8 means 8 omp-threads per task
```

check the queue status:
```
squeue | grep " test "
```
