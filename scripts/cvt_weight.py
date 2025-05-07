import imageio
import os
from glob import glob
import os.path as osp
import shutil
import yaml

src_dir = '/is/cluster/fast/yye/result_hawc/scaleup/x8_EGO-MIX_ftX8_shareTrue'
# dst_dir = '/is/cluster/fast/yye/result_hawc/release/mix_all'
dst_dir = 'output/release/mix_all'



def cvt_weight():
    # copy src_dir/checkpoints/last.ckpt to dst_dir/checkpoints/last.ckpt
    os.makedirs(os.path.join(dst_dir, 'checkpoints'), exist_ok=True)
    shutil.copy(os.path.join(src_dir, 'checkpoints', 'last.ckpt'), os.path.join(dst_dir, 'checkpoints', 'last.ckpt'))
    

    # load yaml
    with open(os.path.join(src_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    # modify config
    config['MODEL']['TARGET'] = 'HAPTIC'
    config['paths']['log_dir'] = 'output'
    config['expname'] = '/'.join(dst_dir.split('/')[-2:])
    
    # save with good indent
    with open(os.path.join(dst_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, indent=4)

    return 


def cvt_video():
    inp_dir = '/lustre/fast/fast/yye/data/custom'
    dst_dir = 'assets/examples/'
    video_list = yaml.safe_load(open('assets/examples/video_list.yaml', 'r'))
    inp_list = [osp.join(inp_dir, e) for e in video_list]

    # inp_list = glob(osp.join(inp_dir, '*/det.pkl'))
    # inp_list = [osp.dirname(e) for e in inp_list]
    
    for inp_dir in inp_list:
        image_list = sorted(glob(osp.join(inp_dir, '*.*g')))

        image_list = [imageio.imread(e) for e in image_list]
        
        index = osp.basename(inp_dir)
        dst_file = osp.join(dst_dir, index, 'video.mp4')
        os.makedirs(osp.dirname(dst_file), exist_ok=True)
        imageio.mimwrite(dst_file, image_list, fps=30)
        print(f'write {dst_file}')        


def cvt_model():
    from demo import load_haptic_model
    ckpt_path = osp.join(dst_dir, 'checkpoints', 'last.ckpt')
    model = load_haptic_model(ckpt_path, device='cuda:0')


def zip_training_data():
    data_parent = '/lustre/fast/fast/yye/data/'
    dst_parent = '/lustre/fast/fast/yye/data/haptic_training_label/'
    os.makedirs(dst_parent, exist_ok=True)
    # clip_dir = sorted(glob(osp.join(data_parent, '*/clip/*data.pyd')))
    clip_dir = sorted(glob(osp.join(data_parent, 'dexycb/clip/*data.pyd')))
    clip_dir = [osp.dirname(e) for e in clip_dir]



    clip_dir_list = list(set(clip_dir))
    for clip_dir in clip_dir_list:
        index = clip_dir.split('/')[-2]
        print(clip_dir)

        # zip clip_dir/ to osp.join(dst_parent, index + '.tar')
        cmd = f"tar -czf {osp.join(dst_parent, index + '.tar')} -C {data_parent} {index}/clip/"
        print(cmd)
        os.system(cmd)
        


if __name__ == '__main__':
    # cvt_weight()
    # cvt_video()
    # cvt_model()

    zip_training_data()

    print('done')