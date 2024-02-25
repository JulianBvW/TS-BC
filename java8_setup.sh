#!/bin/bash

# must be run with ". script.sh"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so

#if [[ $LD_LIBRARY_PATH != *"/usr/lib/nvidia"* ]]; then
	echo "Performing command:  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
#fi

#if [[ $LD_LIBRARY_PATH != *"/usr/local/nvidia/lib64"* ]]; then
	echo "Performing command:  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/nvidia/lib64"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/nvidia/lib64"
#fi

#if [[ $PATH != *"/usr/lib/jvm/java-8-openjdk-amd64/bin/"* ]]; then
	echo "Performing command:  export PATH=/usr/lib/jvm/java-8-openjdk-amd64/bin/:\$PATH"
	export PATH="/usr/lib/jvm/java-8-openjdk-amd64/bin/:$PATH"
#fi

if java -version 2>&1 | grep -q "1.8"; then
	echo "Java 8 is in use."
else
	echo "Java 8 is NOT in use."
fi
