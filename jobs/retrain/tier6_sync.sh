#!/bin/bash
# RUN THIS ON: your Mac, after squeue on the cluster shows tier 6_fit.sh's jobs COMPLETED.
# Tier 6 sync -- 50 fit jobs -> 25 run folders.
# This is the FULLY EXPLICIT version: the filter rules below are written out literally (not
# computed at runtime), so you can read exactly what will and won't be copied before running it.
# Safe to copy-paste this whole block directly into your own terminal instead of running this
# file, if you'd rather not run any script of mine against your cluster account.

FILTER_FILE=$(mktemp)
cat > "$FILTER_FILE" <<'FILTEREOF'
+ /clean_dnn_env_terrain_set2_plot_level_seed42/
+ /clean_dnn_env_terrain_set2_plot_level_seed42/***
+ /clean_dnn_env_terrain_set2_plot_level_seed43/
+ /clean_dnn_env_terrain_set2_plot_level_seed43/***
+ /clean_dnn_env_terrain_set2_plot_level_seed44/
+ /clean_dnn_env_terrain_set2_plot_level_seed44/***
+ /clean_dnn_env_terrain_set2_plot_level_seed45/
+ /clean_dnn_env_terrain_set2_plot_level_seed45/***
+ /clean_dnn_env_terrain_set2_plot_level_seed46/
+ /clean_dnn_env_terrain_set2_plot_level_seed46/***
+ /clean_dnn_noenv_plot_level_seed42/
+ /clean_dnn_noenv_plot_level_seed42/***
+ /clean_dnn_noenv_plot_level_seed43/
+ /clean_dnn_noenv_plot_level_seed43/***
+ /clean_dnn_noenv_plot_level_seed44/
+ /clean_dnn_noenv_plot_level_seed44/***
+ /clean_dnn_noenv_plot_level_seed45/
+ /clean_dnn_noenv_plot_level_seed45/***
+ /clean_dnn_noenv_plot_level_seed46/
+ /clean_dnn_noenv_plot_level_seed46/***
+ /clean_pinn_env_terrain_k_set2_plot_level_seed42/
+ /clean_pinn_env_terrain_k_set2_plot_level_seed42/***
+ /clean_pinn_env_terrain_k_set2_plot_level_seed43/
+ /clean_pinn_env_terrain_k_set2_plot_level_seed43/***
+ /clean_pinn_env_terrain_k_set2_plot_level_seed44/
+ /clean_pinn_env_terrain_k_set2_plot_level_seed44/***
+ /clean_pinn_env_terrain_k_set2_plot_level_seed45/
+ /clean_pinn_env_terrain_k_set2_plot_level_seed45/***
+ /clean_pinn_env_terrain_k_set2_plot_level_seed46/
+ /clean_pinn_env_terrain_k_set2_plot_level_seed46/***
+ /clean_pinn_env_terrain_set2_plot_level_seed42/
+ /clean_pinn_env_terrain_set2_plot_level_seed42/***
+ /clean_pinn_env_terrain_set2_plot_level_seed43/
+ /clean_pinn_env_terrain_set2_plot_level_seed43/***
+ /clean_pinn_env_terrain_set2_plot_level_seed44/
+ /clean_pinn_env_terrain_set2_plot_level_seed44/***
+ /clean_pinn_env_terrain_set2_plot_level_seed45/
+ /clean_pinn_env_terrain_set2_plot_level_seed45/***
+ /clean_pinn_env_terrain_set2_plot_level_seed46/
+ /clean_pinn_env_terrain_set2_plot_level_seed46/***
+ /clean_pinn_noenv_plot_level_seed42/
+ /clean_pinn_noenv_plot_level_seed42/***
+ /clean_pinn_noenv_plot_level_seed43/
+ /clean_pinn_noenv_plot_level_seed43/***
+ /clean_pinn_noenv_plot_level_seed44/
+ /clean_pinn_noenv_plot_level_seed44/***
+ /clean_pinn_noenv_plot_level_seed45/
+ /clean_pinn_noenv_plot_level_seed45/***
+ /clean_pinn_noenv_plot_level_seed46/
+ /clean_pinn_noenv_plot_level_seed46/***
- /growth_curve_attribution/gnnwr/models/
- *
FILTEREOF

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --filter="merge $FILTER_FILE" \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

rm -f "$FILTER_FILE"
