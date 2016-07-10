#!/bin/bash


echo "#include \"testing.h\"" >all.cc

for i in `seq 1 500`;
do
    echo $i
    echo "#define NR \"$i\"" >> all.cc
    cat base.cc_ >>all.cc
    echo "#undef NR" >>all.cc
done

