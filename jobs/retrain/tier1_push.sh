#!/bin/bash
# RUN THIS ON: your Mac, after tier1_fit.sh has locally fit baselines for every split.
# Tier 1 PUSH (upload, Mac -> cluster) -- every PINN fit job needs this file present on
# whichever machine IT runs on (the cluster). Pushes just the 4 chapman_richards params.json
# folders (one per split, plus the 5 spatial_block_kfold fold variants) -- nothing else.
# FULLY EXPLICIT: filter rules written out literally, safe to copy-paste into your own terminal.

FILTER_FILE=$(mktemp)
cat > "$FILTER_FILE" <<'FILTEREOF'
+ /chapman_richards/
+ /chapman_richards/***
+ /spatial_block/
+ /spatial_block/chapman_richards/
+ /spatial_block/chapman_richards/***
+ /temporal/
+ /temporal/chapman_richards/
+ /temporal/chapman_richards/***
+ /spatial_block_kfold/
+ /spatial_block_kfold/chapman_richards_fold0/
+ /spatial_block_kfold/chapman_richards_fold0/***
+ /spatial_block_kfold/chapman_richards_fold1/
+ /spatial_block_kfold/chapman_richards_fold1/***
+ /spatial_block_kfold/chapman_richards_fold2/
+ /spatial_block_kfold/chapman_richards_fold2/***
+ /spatial_block_kfold/chapman_richards_fold3/
+ /spatial_block_kfold/chapman_richards_fold3/***
+ /spatial_block_kfold/chapman_richards_fold4/
+ /spatial_block_kfold/chapman_richards_fold4/***
- *
FILTEREOF

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --filter="merge $FILTER_FILE" \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/ \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/

rm -f "$FILTER_FILE"
