#! /bin/bash

g++ `pkg-config -cflags opencv4` `pkg-config -libs opencv4` -Iinclude -o detectTorso src/*.cpp