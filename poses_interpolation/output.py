from colmap2poses import hello
from inter_poses import main
def getCamsCenter(data_dir):
    # 从colmap生成cams_meta.npy
    main(data_dir)
    # 对位姿插帧，生成所需格式的观看视角参数
    return hello(data_dir)
