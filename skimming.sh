#!/bin/bash

echo "Starting job on $(hostname)"
source /cvmfs/cms.cern.ch/cmsset_default.sh

# Optional: activate your CMSSW environment if needed
# cd /path/to/CMSSW/src
# cmsenv

# Create output directory (inside Condor sandbox)
mkdir -p output

# Run your script
python3 /data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/skim_rootfiles.py

echo "Skimming completed."