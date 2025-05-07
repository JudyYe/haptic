# TODO
# [x] change visualization to sync video

# [x] change dir
# [x] download model
# [x] download demo data 
# install and environment

# evaluation code

# [x] download training data

# readme

python -m train -m \
    expname=reproduce/\${DATASETS.name} \
    data=ego_mix \
    trainer=ddp  \


python -m demo -m    \
   expname=release/mix_all \
   data=custom \

