#!/bin/bash

echo "Starting job on $(hostname)"
echo "Job index: $1"

# Source the filelist (must contain a proper bash array declaration)
source filelists.txt

# Pick the file for this job index
INDEX=$1
INPUT=${FILES[$INDEX]}

echo "Processing input file: $INPUT"

# Set up the CMS environment (adjust as needed)
source /cvmfs/cms.cern.ch/cmsset_default.sh

# Run the skimming script
python skim.py "$INPUT"

echo "Job completed for $INPUT"