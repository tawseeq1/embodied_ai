#!/bin/bash
#SBATCH -J generate_prompts            #Job name(--job-name)
#SBATCH -o %j.out_          #Name of stdout output file(--output)
#SBATCH -e %j.err_          #Name of stderr error file(--error)
#SBATCH -p shared              #Queue (--partition) name ; available options "shared,medium,large or gpu" 
#SBATCH -n 3                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH -c 2                    #(--cpus-per-task) Number of Threads
#SBATCH -t 02:30:00         # specifies walltime(--time maximum duration)of run
#SBATCH --mail-user=tawseeq@kgpian.iitkgp.ac.in        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job

module load compiler/cuda/11.7
conda activate pyt
python3 /scratch/22ch10090/visionlanguage/inria/generate_prompts_gemini.py

