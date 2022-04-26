#!/bin/bash

HERE_DIR=`cd "\`dirname \"$0\"\`";pwd`
N=16

if test $# -ne 2; then
    echo "Usage: $0 input-dir output-dir"
    exit 1
fi

input_dir="$1"
output_dir="$2"

if test -e "$output_dir"; then
    echo "$output_dir already exists, not overwriting"
else
    cp -r "$input_dir" "$output_dir"
fi

cd "$output_dir"

echo "Generating aliased.png images.."
for jf in *.json; do
    ((i=i%N)); ((i++==0)) && wait
    {
        if ! test -f "${jf%.json}.aliased.png"; then
            python3 "${HERE_DIR}/../generate_plots/generate_aliased_from_labels.py" "$jf" 
        fi
        if ! test -f "$jf".orig; then
            mv "$jf" "$jf".orig
        fi
        echo -n '.'
    } &
done
wait

echo "Generating new json files.."
for f in *.aliased.png; do
    ((i=i%N)); ((i++==0)) && wait
    {
        if ! test -f "${f%.aliased.png}.json"; then
            "${HERE_DIR}/../generate_plots/gt_from_pairs/build/gt_from_pairs" "${f%.aliased.png}"
        fi
        echo -n '.'        
    } &
done
wait

echo 'Done.'
