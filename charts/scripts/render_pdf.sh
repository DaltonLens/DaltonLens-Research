#!/bin/bash

if test $# -ne 1; then
    echo "Usage: $0 pdf_file"
    exit 1
fi

# apt-get install mupdf-tools cairosvg poppler-utils

# WARNING: the gs output might not have the exact same size :(
mutool draw -O text=path -o "$1".svg "$1" 1
cairosvg "$1".svg -o "$1"_textpath.pdf

# gs is the only one really filling any pixel that gets touched by a path.
gs -r72 -dNOPAUSE -dBATCH -sDEVICE=png16m -sOutputFile="$1.r72.antialiased.png" -dGraphicsAlphaBits=4 -dTextAlphaBits=1 "$1"_textpath.pdf
gs -r72 -dNOPAUSE -dBATCH -sDEVICE=png16m -sOutputFile="$1.r72.aliased.png" -dGraphicsAlphaBits=1 -dTextAlphaBits=1 "$1"_textpath.pdf

gs -r56 -dNOPAUSE -dBATCH -sDEVICE=png16m -sOutputFile="$1.r56.antialiased.png" -dGraphicsAlphaBits=4 -dTextAlphaBits=1 "$1"_textpath.pdf
gs -r56 -dNOPAUSE -dBATCH -sDEVICE=png16m -sOutputFile="$1.r56.aliased.png" -dGraphicsAlphaBits=1 -dTextAlphaBits=1 "$1"_textpath.pdf

# 72 dpi is the default for pdf. This will keep the page size unchanged.
# scale_to=`identify -format '-scale-to-x %w -scale-to-y %h' "$1.aliased_gs.png"`
# pdftoppm -r 72 -aa yes -aaVector yes -png "$1"_textpath.pdf > "$1.antialiased.png"
# pdftoppm -r 72 -thinlinemode solid -aa no -aaVector no -png "$1"_textpath.pdf > "$1.aliased.png"
