#!/bin/bash
#SBATCH -J parallel_dmercator
#SBATCH -c 8 # Number of cores requested
#SBATCH -t 4000 # Runtime in minutes
#SBATCH -p mweber_compute # Partition to submit to, mweber_gpu
# #SBATCH --gres=gpu:1
#SBATCH --mem=64000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o test_%j.out # Standard out goes to this file
#SBATCH -e test_%j.err # Standard err goes to this filehostname

module load gcc/12.2.0-fasrc01
module load cmake
module load cuda/12.9.1-fasrc01

nvidia-smi || true

# Build project
chmod +x build.sh
rm -r build-cuda/
cmake -S . -B build-cuda -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DDMERCATOR_ENABLE_CUDA=ON
cmake --build build-cuda -j 8
#./build.sh -b Release

# Run embeddings
srun ./mercator -d 1 /n/holylabs/LABS/mweber_lab/Everyone/rjankowski/repo/d-mercator/output/email/email_GC.edge
