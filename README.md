compile the source on the remote host:

ssh compiler
module load impi
mpicxx floyd.cpp -o floyd
exit


run the binary from home directory:

module load impi # you can run it only once during the session
cp floyd ./_scratch
sbatch -p test -n32 impi ~/floyd input-example.txt # n32 means 32 threads


check the queue status:
squeue | grep " test "
