#!/bin/bash
# RUN THIS ON: your Mac, after squeue on the cluster shows tier 5_fit.sh's jobs COMPLETED.
# Tier 5 sync -- 30 fit jobs -> 15 run folders.
# This is the FULLY EXPLICIT version: the filter rules below are written out literally (not
# computed at runtime), so you can read exactly what will and won't be copied before running it.
# Safe to copy-paste this whole block directly into your own terminal instead of running this
# file, if you'd rather not run any script of mine against your cluster account.

FILTER_FILE=$(mktemp)
cat > "$FILTER_FILE" <<'FILTEREOF'
+ /temporal/
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed42/
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed42/***
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed43/
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed43/***
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed44/
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed44/***
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed45/
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed45/***
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed46/
+ /temporal/clean_dnn_env_terrain_set2_temporal_seed46/***
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed42/
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed42/***
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed43/
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed43/***
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed44/
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed44/***
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed45/
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed45/***
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed46/
+ /temporal/clean_pinn_env_terrain_k_set2_temporal_seed46/***
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed42/
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed42/***
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed43/
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed43/***
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed44/
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed44/***
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed45/
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed45/***
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed46/
+ /temporal/clean_pinn_env_terrain_set2_temporal_seed46/***
- /growth_curve_attribution/gnnwr/models/
- *
FILTEREOF

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --filter="merge $FILTER_FILE" \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

rm -f "$FILTER_FILE"
