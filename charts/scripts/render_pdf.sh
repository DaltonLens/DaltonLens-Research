#!/bin/bash

if test $# -ne 1; then
    echo "Usage: $0 pdf_file"
    exit 1
fi

# 72 dpi is the default for pdf. This will keep the page size unchanged.
pdftoppm -r 72 -thinlinemode shape -aa no -aaVector no -png "$1" > "$1_aliased.png"
pdftoppm -r 72 -aa yes -aaVector yes -png "$1" > "$1_antialiased.png"
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -sOutputFile="$1_antialiased_gs.png" -dGraphicsAlphaBits=1 -dTextAlphaBits=1 "$1"
