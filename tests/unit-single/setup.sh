#!/bin/bash


cp base.cc_1 all.cc

for i in `seq 1 500`;
do
    echo "#define NR \"$i\"" >>all.cc
    cat base.cc_2 >>all.cc
    echo "#undef NR" >>all.cc
done
