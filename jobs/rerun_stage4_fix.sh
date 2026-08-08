#!/bin/bash
# Re-run of the 30 stage4_all_environmental fit jobs, generated from
# jobs/submit_experiments.py::build_stage_sweep_jobs() to guarantee exact argument match.
# Run once: bash rerun_stage4_fix.sh
set -e

sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 0
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 0
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 0
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 1
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 1
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 1
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 2
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 2
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 2
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 3
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 3
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 3
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 4
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 4
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 4
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 0
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 0
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 0
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 1
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 1
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 1
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 2
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 2
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 2
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 3
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 3
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 3
sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 40 spatial_block_kfold 42 stage4_all_environmental_dnn_env_terrain 256 stage4_all_environmental 0.0 42 5 4
sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain 42 256 stage4_all_environmental 0.0 42 5 4
sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 6survey 500 40 spatial_block_kfold 1.0 1.0 stage4_all_environmental_pinn_env_terrain_k 42 256 stage4_all_environmental 0.0 42 5 4
