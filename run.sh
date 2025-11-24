#!/bin/bash

# ================= CONFIGURATION =================
DATA_DIR="DATA"
BF_CUTOFF=60    # 60 seconds for Brute Force
LS_CUTOFF=60    # 60 seconds for Local Search
LS_SEEDS=10     # Number of seeds to run for LS
PYTHON_CMD="python" # Or python3
# =================================================

# 1. Initialize CSV with Headers
echo "Dataset,BF Time,BF Qual,BF Full?,App Time,App Qual,App Err,LS Time,LS Qual,LS Err" > results.csv
echo "--------------------------------------------------------"
echo "Starting Experiments..."
echo "Results will be saved to results.csv"
echo "--------------------------------------------------------"

# 2. Loop through all .tsp files
for inst in "$DATA_DIR"/*.tsp; do
    [ -e "$inst" ] || continue
    
    base_name=$(basename "$inst")
    echo "Processing $base_name..."

    # ==========================================
    # A. BRUTE FORCE (BF)
    # ==========================================
    echo "  > Running Brute Force..."
    # Capture output line containing [RESULT]
    output=$($PYTHON_CMD run.py -inst "$inst" -alg BF -time $BF_CUTOFF -seed 0 | grep "\[RESULT\]")
    
    # Parse the values from string: [RESULT] time=0.123 cost=1000 status=COMPLETED
    bf_time=$(echo $output | awk -F'time=' '{print $2}' | awk '{print $1}')
    bf_cost=$(echo $output | awk -F'cost=' '{print $2}' | awk '{print $1}')
    bf_stat=$(echo $output | awk -F'status=' '{print $2}' | awk '{print $1}')
    
    # Convert status to Yes/No for "Full Tour" column
    if [ "$bf_stat" == "COMPLETED" ]; then
        bf_full="Yes"
    else
        bf_full="No"
    fi

    # ==========================================
    # B. APPROXIMATION (Approx)
    # ==========================================
    echo "  > Running Approx..."
    output=$($PYTHON_CMD run.py -inst "$inst" -alg Approx -time 0 -seed 0 | grep "\[RESULT\]")
    
    app_time=$(echo $output | awk -F'time=' '{print $2}' | awk '{print $1}')
    app_cost=$(echo $output | awk -F'cost=' '{print $2}' | awk '{print $1}')

    # ==========================================
    # C. LOCAL SEARCH (LS) - 10 Runs
    # ==========================================
    echo -n "  > Running LS ($LS_SEEDS seeds): "
    
    ls_costs=()
    ls_total_cost=0
    
    # Loop seeds 1 to 10
    for (( i=1; i<=$LS_SEEDS; i++ )); do
        output=$($PYTHON_CMD run.py -inst "$inst" -alg LS -time $LS_CUTOFF -seed $i | grep "\[RESULT\]")
        cost=$(echo $output | awk -F'cost=' '{print $2}' | awk '{print $1}')
        
        ls_costs+=($cost)
        echo -n "."
    done
    echo " Done."

    # ==========================================
    # D. CALCULATIONS (Using Python for safety)
    # ==========================================
    
    # Pass all LS costs to python to calculate avg and min (best)
    # We use python -c because bash math is terrible with floats
    
    # Convert array to comma-separated string
    ls_costs_str=$(IFS=,; echo "${ls_costs[*]}")
    
    read ls_avg_qual ls_best_qual <<< $(python -c "
costs = [$ls_costs_str]
avg_c = sum(costs) / len(costs)
min_c = min(costs)
print(f'{avg_c} {min_c}')
")

    # Calculate Relative Errors
    # RelError = (Value - Best_LS) / Best_LS
    # Note: If BF found a better solution than LS (unlikely for large N), 
    # the rubric says 'relative to the best solution YOUR LS finds'.
    
    read app_err ls_err <<< $(python -c "
base = $ls_best_qual
app_c = $app_cost
ls_avg = $ls_avg_qual

if base == 0: base = 1 # Avoid div/0

app_e = (app_c - base) / base
ls_e = (ls_avg - base) / base

print(f'{app_e:.4f} {ls_e:.4f}')
")

    # ==========================================
    # E. SAVE TO CSV
    # ==========================================
    # Format: Dataset, BF_Time, BF_Qual, BF_Full, App_Time, App_Qual, App_Err, LS_Time, LS_Qual, LS_Err
    
    echo "$base_name,$bf_time,$bf_cost,$bf_full,$app_time,$app_cost,$app_err,$LS_CUTOFF,$ls_avg_qual,$ls_err" >> results.csv

done

echo "--------------------------------------------------------"
echo "Done! Check results.csv"