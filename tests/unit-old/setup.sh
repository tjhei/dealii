#!/bin/bash

for i in `seq 1 500`;
do
    cp base_test.cc test_$i.cc
    cp base_test.debug.output1 test_$i.debug.output
done
