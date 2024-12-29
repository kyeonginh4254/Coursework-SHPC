#!/bin/bash

: ${NODES:=4}

TMPDIR=~ salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		nsys profile --cudabacktrace=all \
		./main -n 16384 -v $@