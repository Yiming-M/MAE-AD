#!/bin/bash
export NCCL_P2P_DISABLE=1 && export NCCL_IB_DISABLE=1;
python main.py --category zipper
