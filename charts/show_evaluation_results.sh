#!/bin/bash

generate_filelist() {        
    while read f; do
        for d in logs/*/*/evaluation; do 
            echo "$d/$f"
        done
    done < test_images.txt
}

generate_filelist | xargs -d '\n' zv-python
