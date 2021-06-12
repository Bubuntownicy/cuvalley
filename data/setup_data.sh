#!/bin/sh

for compressed in *.tar.gz; do
    if [ ! -f "${compressed%.*.*}" ]; then
        tar -xzf $compressed
        for file in "${compressed%.*.*}"/*;do
            python3 pickle_file.py "${compressed%.*.*}" `basename $file`
        done
    fi
done
