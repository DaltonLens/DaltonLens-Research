#!/bin/bash

logdir="$1"
if test -z $logdir; then
    logdir="logs"
fi

for d in "$logdir"/*/*/evaluation; do
    echo $d
    cat "$d/evaluation.txt"
    echo
done

generate_filelist() {        
    while read f; do
        for d in "$logdir"/*/*/evaluation; do 
            echo "$d/$f"
        done
    done < test_images.txt
}

generate_filelist | xargs -d '\n' zv
