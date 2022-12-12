#!/bin/bash
#PBS -q qgpu
#PBS -A DD-22-68
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -l nsys=True
#PBS -N PCG-NBODY-OPENACC

profile="/apps/all/CUDAcore/11.1.1/nsight-compute-2020.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu --force-overwrite  --target-processes application-only  --replay-mode kernel  --kernel-regex-base function  --launch-skip-before-match 0  --section ComputeWorkloadAnalysis  --section InstructionStats  --section LaunchStats  --section MemoryWorkloadAnalysis  --section MemoryWorkloadAnalysis_Chart  --section MemoryWorkloadAnalysis_Tables  --section Occupancy  --section SchedulerStats  --section SourceCounters  --section SpeedOfLight  --section SpeedOfLight_RooflineChart  --section WarpStateStats  --sampling-interval auto  --sampling-max-passes 5  --sampling-buffer-size 33554432  --profile-from-start 1  --cache-control all  --clock-control base  --apply-rules yes  --check-exit-code yes"

date

PROJECT_DIR=$PBS_O_WORKDIR

ml NVHPC/22.2
ml HDF5/1.12.1-NVHPC-22.2


STEP=step1
echo $STEP
cd $PROJECT_DIR/$STEP
make
make run
make check_output



for i in {10..25}
do
    n=$(bc <<< "5 * $i * 512")
    ./nbody $n 0.01f 500 20 ../commons/$n.h5 /dev/null
    $profile --export ./profile$i ./nbody $n 0.01f 500 20 ../commons/$n.h5 /dev/null
done

cd $PROJECT_DIR/tests
python3 -m venv py-test-env

source py-test-env/bin/activate
python3 -m pip install h5py
./run_tests.sh $PROJECT_DIR/$STEP


date