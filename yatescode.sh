#!/bin/sh
topo=$1
#move the demand file to the directory where you can run yates
mv demand_${topo}.txt ~/yates/data/demands/
#move to the directory where you can run yates
alias proj="cd ~/yates"
#yates stores generated files in /data/results/abilene/. Create it if it does not exist
mkdir -p ~/yates/data
mkdir -p ~/yates/data/results
mkdir -p ~/yates/data/results/${topo}

#yates command that generates a file called semimcfraeke_0
#with all the paths between all the possible source-destination pairs
yates ~/yates/data/topologies/${topo}.dot ~/yates/data/demands/demand_${topo}.txt ~/yates/data/demands/demand_${topo}.txt ~/yates/data/hosts/${topo}.hosts -semimcfraeke >> outputyates.txt

#~/yates/data/demands/3cycle_demands.txt ~/yates/data/demands/3cycle_demands.txt   ~/yates/data/hosts/${topo}.hosts -semimcfraeke
