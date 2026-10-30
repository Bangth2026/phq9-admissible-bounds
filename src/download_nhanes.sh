#!/usr/bin/env bash
# Download the NHANES components used in the study.
set -e
mkdir -p data && cd data
B=https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public
for f in DEMO DPQ SLQ BMX HUQ SMQ PAQ MCQ DIQ BPQ; do
  curl -sfO "$B/2017/DataFiles/P_$f.xpt"
  curl -sfO "$B/2021/DataFiles/${f}_L.xpt"
  for y in "2005 D" "2007 E" "2009 F" "2011 G" "2013 H" "2015 I"; do set -- $y
    curl -sfO "$B/$1/DataFiles/${f}_$2.xpt" || true
  done
done
