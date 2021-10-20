import argparse

from modules import *
from functions import *


# Description:
# Denoising a benchmark set of images  
def process_image_set_func():
    opt = parse_options()

    torch.manual_seed(opt.seed)
    if opt.gpu_usage > 0 and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    sys.stdout = Logger('./logs/process_image_set_log.txt')

    clean_im_list, im_name_list = read_image_set(opt.in_folder)

    noisy_im_list = list()
    for i in range(len(clean_im_list)):
        noisy_im_list.append(clean_im_list[i] + (opt.sigma / 255) * torch.randn_like(clean_im_list[i]))

    denoised_im_list, denoising_time_list = denoise_image_list(noisy_im_list, im_name_list, \
        opt.sigma, opt.gpu_usage, opt.silent)

    denoised_psnr_list = list()
    for i in range(len(clean_im_list)):
        denoised_im_list[i] = denoised_im_list[i].squeeze(0).detach()
        clean_im_list[i] = clean_im_list[i].squeeze(2).squeeze(0)
        noisy_im_list[i] = noisy_im_list[i].squeeze(2).squeeze(0)
        denoised_psnr_tmp = -10 * math.log10(((denoised_im_list[i] - clean_im_list[i]) ** 2).mean())
        denoised_psnr_list.append(denoised_psnr_tmp)

    print('image set done, psnr: {:.4f}'.format(np.array(denoised_psnr_list).mean()))
    sys.stdout.flush()

    if opt.save:
        for i in range(len(denoised_im_list)):
            (im_name, im_ext) = os.path.splitext(im_name_list[i])
            save_image(noisy_im_list[i], opt.out_folder, '{}_noisy_{}{}'.format(im_name, opt.sigma, im_ext))
            save_image(denoised_im_list[i], opt.out_folder, '{}_denoised_{}{}'.format(im_name, opt.sigma, im_ext))

    return


# Description:
# Parsing command line
# 
# Outputs:
# opt - options
def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_folder', type=str, default='./data_set/cbsd68/', help='path to the input folder')
    parser.add_argument('--out_folder', type=str, default='./output/images/set/', help='path to the output folder')
    parser.add_argument('--sigma', type=int, default=25, help='noise sigma')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--gpu_usage', type=int, default=0, help='0 - use CPU, 1 - use GPU')
    parser.add_argument('--save', action='store_true', help='save the denoised image')
    parser.add_argument('--silent', action='store_true', help="don't print 'done' every image")

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    process_image_set_func()
