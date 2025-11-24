#!/bin/bash

# ================= CONFIGURATION =================
# We pick one large dataset to demonstrate the effect
DATASET="DATA/Boston.tsp"
# Times to test (in seconds)
TIMES=(10 30 60 120 300)
# We use a fixed seed to ensure differences are due to time, not random luck
SEED=42
PYTHON_CMD="python"
# =================================================

echo "Cutoff Time,Solution Quality" > cutoff_results.csv

echo "------------------------------------------------"
echo "Running Time Analysis on $DATASET"
echo "------------------------------------------------"

for t in "${TIMES[@]}"; do
    echo -n "Running for ${t}s... "
    
    # Run the main script
    output=$($PYTHON_CMD run.py -inst "$DATASET" -alg BF -time $t -seed $SEED | grep "\[RESULT\]")
    
    # Extract cost
    cost=$(echo $output | awk -F'cost=' '{print $2}' | awk '{print $1}')
    
    # Save to CSV
    echo "$t,$cost" >> cutoff_results.csv
    
    echo "Done. Cost: $cost"
done

echo "------------------------------------------------"
echo "Data saved to cutoff_results.csv"