set -x

# download haptic model
gdown 1BX_gT__7hE47B_YUizUWfEfeqopLxloZ  -O output/haptic_model.tar.gz

tar -xvf output/haptic_model.tar.gz -C output/
rm output/haptic_model.tar.gz

