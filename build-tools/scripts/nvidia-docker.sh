#!/bin/bash

if which nvidia-docker >/dev/null
then
    nvidia-docker "$@"
else
    if [ "$1" == "run" ]
    then
        GPU=\"device=${NV_GPU}\" 
        if [ "${NV_GPU}" == "" ]
        then
            GPU=all
        fi
        docker run --gpus=$GPU "${@:2}"
    else
        docker "$@"
    fi
fi

