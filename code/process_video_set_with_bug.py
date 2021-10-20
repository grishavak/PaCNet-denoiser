import argparse
import matplotlib.pyplot as plt
import shutil

from modules import *
from functions import *


# Description:
# Denoising a benchmark set of video sequences  
def process_video_set_with_bug_func():
    opt = parse_options()

    torch.manual_seed(opt.seed)
    if opt.gpu_usage > 0 and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    clip_str = 'clip_' if opt.clipped_noise else ''
    sys.stdout = Logger('./logs/process_video_set_with_bug_{}{}_log.txt'.format(clip_str, opt.sigma))

    video_names = sorted(os.listdir(opt.in_folder))
    denoised_psnr_list = list()
    for i in range(len(video_names)):
        vid_name = video_names[i]
        vid_folder = opt.in_folder + '{}/'.format(vid_name)
        clean_vid = read_video_sequence(vid_folder, opt.max_frame_num, opt.file_ext)

        denoised_vid_t, denoised_vid_s, denoising_time = denoise_video_sequence_with_bug(clean_vid, vid_name, \
            opt.sigma, opt.clipped_noise, opt.gpu_usage, opt.silent)
    
        denoised_vid_t = denoised_vid_t.squeeze(0).detach()
        clean_vid = clean_vid.squeeze(0)
        denoised_psnr_t = -10 * torch.log10(((denoised_vid_t - clean_vid) ** 2).mean(dim=(-4, -2, -1), keepdim=False))
        denoised_psnr_s = -10 * torch.log10(((denoised_vid_s - clean_vid) ** 2).mean(dim=(-4, -2, -1), keepdim=False))
        denoised_psnr_list.append(denoised_psnr_t.mean())

        print('')
        print('-' * 80)
        print('{}/{}: sequence {} done (with bug), psnr: {:.2f}, denoising time: {:.2f} ({:.2f} per frame)'.\
            format(i + 1, len(video_names), vid_name.upper(), denoised_psnr_t.mean(), 
            denoising_time, denoising_time / clean_vid.shape[1]))
        print('-' * 80)
        print('')
        sys.stdout.flush()

        if opt.save_jpg:
            denoised_folder_jpg = opt.jpg_out_folder + '/denoised_with_bug_{}{}/'.format(clip_str, opt.sigma)
            if not os.path.exists(denoised_folder_jpg):
                os.mkdir(denoised_folder_jpg)

            denoised_sequence_folder = denoised_folder_jpg + vid_name + '/'
            if os.path.exists(denoised_sequence_folder):
                shutil.rmtree(denoised_sequence_folder)
            os.mkdir(denoised_sequence_folder)

            for i in range(denoised_vid_t.shape[1]):
                save_image(denoised_vid_t[:, i, ...], denoised_sequence_folder, '{:05}.jpg'.format(i))

        if opt.save_avi:
            denoised_folder_avi = opt.avi_out_folder + '/denoised_with_bug_{}{}/'.format(clip_str, opt.sigma)
            if not os.path.exists(denoised_folder_avi):
                os.mkdir(denoised_folder_avi)

            save_video_avi(denoised_vid_t, denoised_folder_avi, vid_name)

    print('')
    print('-' * 60)
    print('Denoising with bug with sigma {} done, average psnr: {:.2f}'.\
        format(opt.sigma, np.array(denoised_psnr_list).mean()))
    print('-' * 60)
    print('')
    sys.stdout.flush()

    return


# Description:
# Parsing command line
# 
# Outputs:
# opt - options
def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_folder', type=str, default='./data_set/davis_color/test_dev_seq/', help='path to the input folder')
    parser.add_argument('--file_ext', type=str, default='jpg', help='file extension: {jpg, png}')
    parser.add_argument('--jpg_out_folder', type=str, default='./output/videos/jpg_sequences/set/', \
        help='path to the output folder for JPG frames')
    parser.add_argument('--avi_out_folder', type=str, default='./output/videos/avi_files/set/', \
        help='path to the output folder for AVI files')
    parser.add_argument('--sigma', type=int, default=30, help='noise sigma')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--clipped_noise', type=int, default=0, help='0: AWGN, 1: clipped Gaussian noise')
    parser.add_argument('--gpu_usage', type=int, default=0, \
        help='0 - use CPU, 1 - use GPU for nearest neighbor search, \
              2 - use GPU for whole processing (requires large GPU memory)')
    parser.add_argument('--save_jpg', action='store_true', help='save the denoised video as JPG frames')
    parser.add_argument('--save_avi', action='store_true', help='save the denoised video as AVI file')
    parser.add_argument('--silent', action='store_true', help="don't print 'done' every frame")
    parser.add_argument('--max_frame_num', type=int, default=85, help='maximum number of frames')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    process_video_set_with_bug_func()
