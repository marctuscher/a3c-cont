#!/bin/bash

now=$(date +"%d-%m-%Y-%H-%M")


if [ "$1" = "train" ];
then
    python main.py
elif [ "$1" = "train+" ];
then
    if [ -d "~/old" ];
    then
    echo existing
    cd ~/old
    mkdir $now
    else
    mkdir ~/old
    cd ~/old
    mkdir $now
    fi
    cd ~/festium
    if [ -d "log" ]; then
    mv log ~/old/$now/log/
    fi
    if [ -d "net" ]; then
    mv net ~/old/$now/net/
    fi
    if [ -d "logs" ]; then
    mv logs ~/old/$now/logs
    fi
    python main.py
elif [ "$1" = "store" ];
then
    if [ -d "~/old" ];
    then
    echo existing
    cd ~/old
    mkdir $now
    else
    mkdir ~/old
    cd ~/old
    mkdir $now
    fi
    cd ~/festium
    if [ -d "log" ]; then
    mv log ~/old/$now/log/
    fi
    if [ -d "net" ]; then
    mv net ~/old/$now/net/
    fi
    if [ -d "logs" ]; then
    mv logs ~/old/$now/logs
    fi   
elif [ "$1" = "play" ];
then
    python3 replay.py
elif [ "$1" = "del" ];
then
    rm -r log
    rm -r net
    rm -r logs      
fi  

