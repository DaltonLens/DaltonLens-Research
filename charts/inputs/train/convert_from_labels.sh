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
    echo "$output_dir already exists"
    exit 1
fi

cp -r "$input_dir" "$output_dir"

cd "$output_dir"

echo "Generating aliased.png images.."
for jf in *.json; do
    ((i=i%N)); ((i++==0)) && wait
    {
        python3 "${HERE_DIR}/../generate_plots/generate_aliased_from_labels.py" "$jf" 
        mv "$jf" "$jf".orig
        echo -n '.'
    } &
done
wait

echo
echo "Renaming rendered.png -> antialiased.png.."
rename 's/\.rendered.png/.antialiased.png/' *.rendered.png

echo "Generating new json files.."
for f in *.aliased.png; do
    ((i=i%N)); ((i++==0)) && wait
    {
        "${HERE_DIR}/../generate_plots/gt_from_pairs/build/gt_from_pairs" "${f%.aliased.png}"
        echo -n '.'
    } &
done
wait

echo 'Done.'
