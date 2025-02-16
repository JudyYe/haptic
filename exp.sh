python -m train -m \
    expname=dev/\${DATASETS.name}_\${TRAIN.STAGE}X\${TRAIN.IMAGE_BATCH_SIZE}_share\${MODEL.SHARE_BACKBONE} \
    data=ego_mix \
    MODEL.SHARE_BACKBONE=True  \
    TRAIN.IMAGE_BATCH_SIZE=8 \
    TRAIN.BATCH_SIZE=2 

    trainer=ddp  +engine=mpi engine.bid=50 engine.ngpu=2
