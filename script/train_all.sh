#!/bin/bash

# 一键训练所有模型（OSTrack → UGTrack stage1 → UGTrack stage2）
# 使用 nohup 运行，断网不影响：nohup bash train_all.sh > train_all.log 2>&1 &

# set -e        # 任意一步失败（exit code 非 0）脚本会立即退出，不会继续跑后面的无效训练

cd /root/autodl-tmp/UGTrack

echo "========================================="
echo " [1/6] OSTrack vitb_256_mae_32x4_ep300"
echo "========================================="
python tracking/train.py --script ostrack --config vitb_256_mae_32x4_ep300 --save_dir ./output --mode single

echo "========================================="
echo " [2/6] UGTrack stage1 s1_gru_t3_bce05"
echo "========================================="
python tracking/train.py --script ugtrack --config s1_gru_t3_bce05 --save_dir ./output --mode single

echo "========================================="
echo " [3/6] UGTrack stage2 ugtrack_token"
echo "========================================="
python tracking/train.py --script ugtrack --config ugtrack_token --script_prv ugtrack --config_prv s1_gru_t3_bce05 --save_dir ./output --mode single

echo "========================================="
echo " [4/6] UGTrack stage2 ugtrack_token_ce"
echo "========================================="
python tracking/train.py --script ugtrack --config ugtrack_token_ce --script_prv ugtrack --config_prv s1_gru_t3_bce05 --save_dir ./output --mode single

echo "========================================="
echo " [5/6] UGTrack stage2 ugtrack_token_prune"
echo "========================================="
python tracking/train.py --script ugtrack --config ugtrack_token_prune --script_prv ugtrack --config_prv s1_gru_t3_bce05 --save_dir ./output --mode single

echo "========================================="
echo " [6/6] UGTrack stage2 ugtrack_token_prune_ce"
echo "========================================="
python tracking/train.py --script ugtrack --config ugtrack_token_prune_ce --script_prv ugtrack --config_prv s1_gru_t3_bce05 --save_dir ./output --mode single

echo "========================================="
echo " 全部训练完成！"
echo "========================================="
