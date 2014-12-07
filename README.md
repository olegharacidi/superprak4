compile the source on the remote host:

```
ssh compiler
module load impi
mpicxx floyd.cpp -o ~/_scratch/floyd
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
sbatch -p test -N 32 --ntasks-per-node=1 impi ~/floyd_omp input-example.txt # -N 32 means 32 tasks
```

check the queue status:
```
squeue | grep " test "
```
