#!/bin/bash
# RUN THIS ON: your Mac, after squeue on the cluster shows tier 2_fit.sh's jobs COMPLETED.
# Tier 2 sync -- 30 fit jobs -> 15 run folders.
# This is the FULLY EXPLICIT version: the filter rules below are written out literally (not
# computed at runtime), so you can read exactly what will and won't be copied before running it.
# Safe to copy-paste this whole block directly into your own terminal instead of running this
# file, if you'd rather not run any script of mine against your cluster account.

FILTER_FILE=$(mktemp)
cat > "$FILTER_FILE" <<'FILTEREOF'
+ /spatial_block/
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed42/
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed42/***
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed43/
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed43/***
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed44/
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed44/***
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed45/
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed45/***
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed46/
+ /spatial_block/clean_dnn_env_terrain_set2_spatial_block_seed46/***
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed42/
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed42/***
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed43/
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed43/***
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed44/
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed44/***
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed45/
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed45/***
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed46/
+ /spatial_block/clean_pinn_env_terrain_k_set2_spatial_block_seed46/***
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed42/
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed42/***
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed43/
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed43/***
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed44/
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed44/***
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed45/
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed45/***
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed46/
+ /spatial_block/clean_pinn_env_terrain_set2_spatial_block_seed46/***
- /growth_curve_attribution/gnnwr/models/
- *
FILTEREOF

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --filter="merge $FILTER_FILE" \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

rm -f "$FILTER_FILE"
