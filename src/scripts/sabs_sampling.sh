echo "Running Selections for AID1798..."
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python run_sampling_v2.py augment_data.split="$split" augment_data.name="AID1798" augment_data.sampling_method="sabs" &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 2 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

echo "Running Selections for AID2689..."
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python run_sampling_v2.py augment_data.split="$split" augment_data.name="AID2689" augment_data.sampling_method="sabs" &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 2 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

echo "Running Selections for AID463087..."
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python run_sampling_v2.py augment_data.split="$split" augment_data.name="AID463087" augment_data.sampling_method="sabs" &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 2 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

echo "Running Selections for AID485290..."
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python run_sampling_v2.py augment_data.split="$split" augment_data.name="AID485290" augment_data.sampling_method="sabs" &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 2 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

echo "Running Selections for AID488997..."
process_count=0
for split in random_cv1 random_cv2 random_cv3 random_cv4 random_cv5 scaffold_seed1 scaffold_seed2 scaffold_seed3 scaffold_seed4 scaffold_seed5; do
    python run_sampling_v2.py augment_data.split="$split" augment_data.name="AID488997" augment_data.sampling_method="sabs" &
    process_count=$((process_count + 1))
    if [ "$process_count" -ge 2 ]; then
        wait -n
        process_count=$((process_count - 1))
    fi
done
wait

