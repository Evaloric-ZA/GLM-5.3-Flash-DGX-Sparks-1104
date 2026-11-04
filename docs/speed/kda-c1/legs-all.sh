#!/bin/bash
# The four pre-registered legs (RESULTS.md 2026-11-04): OFF, ON, OFF, ON, all with TF_GLM_PROF=1 on image 45a1d17d74d7.
cd /home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
L=docs/speed/kda-c1/leg.sh
$L off1 TF_GLM_PROF=1 TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe
$L on1 TF_GLM_PROF=1
$L off2 TF_GLM_PROF=1 TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe
$L on2 TF_GLM_PROF=1
echo LEGS_DONE
