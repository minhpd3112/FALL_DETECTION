✅ Workflow Hoàn chỉnh
Tối nay:

bash
# SSH vào server
ssh user@server
# Session 1: BlockGCN
tmux new -s blockgcn
cd BlockGCN && conda activate fall_detection
./train_all_streams.sh
# Ctrl+B, D
# Session 2: ProtoGCN
tmux new -s protogcn
cd ProtoGCN && conda activate protogcn
./train_all_streams.sh
# Ctrl+B, D
# Thoát SSH
exit  # Code vẫn chạy! ✅
Ngày mai:

bash
# SSH lại
ssh user@server
# Check progress
tmux attach -t blockgcn  # Xem BlockGCN logs
tmux attach -t protogcn  # Xem ProtoGCN logs