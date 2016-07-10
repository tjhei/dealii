#!/bin/bash

for i in `seq 1 500`;
do
    cp base.cc_ test_$i.cc
done
