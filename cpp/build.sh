#!/bin/sh
CXX=g++
SRC="main.cpp cnpy.cpp"
CXXFLAGS="`pkg-config --cflags zlib` -std=c++11 -Wall -O3"
LDFLAGS=`pkg-config --libs zlib`
TARGET=dynamic_rnn

rm -rf $TARGET
$CXX $CXXFLAGS $SRC $LDFLAGS -o $TARGET
