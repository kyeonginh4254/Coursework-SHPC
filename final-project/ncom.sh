#!/bin/bash

: ${NODES:=1}

TMPDIR=~ salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		ncu -o ncu_report --set full \
		./main -n 1024 -v $@