import argparse
import matplotlib.pyplot as plt

from modules import *
from functions import *


# Description:
# Image denoising demo  
def process_image_func():
    opt = parse_options()

    torch.manual_seed(opt.seed)
    if opt.gpu_usage > 0 and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    sys.stdout = Logger('./logs/process_image_log.txt')

    clean_f_name = opt.in_folder + opt.im_name
    clean_im = read_image(clean_f_name)
    noisy_im = clean_im + (opt.sigma / 255) * torch.randn_like(clean_im)

    denoised_im, denoising_time = denoise_image(noisy_im, opt.sigma, opt.gpu_usage)
    
    denoised_im = denoised_im.squeeze(0).detach()
    clean_im = clean_im.squeeze(2).squeeze(0)
    noisy_im = noisy_im.squeeze(2).squeeze(0)
    denoised_psnr = -10 * math.log10(((denoised_im - clean_im) ** 2).mean())

    print('image {} done, psnr: {:.4f}, denoising time: {:.4f}'.\
        format(os.path.splitext(opt.im_name)[0].upper(), denoised_psnr, denoising_time))
    sys.stdout.flush()

    if opt.save:
        (im_name, im_ext) = os.path.splitext(opt.im_name)
        save_image(noisy_im, opt.out_folder, '{}_noisy_{}{}'.format(im_name, opt.sigma, im_ext))
        save_image(denoised_im, opt.out_folder, '{}_denoised_{}{}'.format(im_name, opt.sigma, im_ext))

    if opt.plot:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(float_tensor_to_np_uint8(clean_im), vmin=0, vmax=255)
        axs[0].set_title('Clean')
        axs[1].imshow(float_tensor_to_np_uint8(noisy_im), vmin=0, vmax=255)
        axs[1].set_title(r'Noisy with $\sigma$ = {}'.format(opt.sigma))
        axs[2].imshow(float_tensor_to_np_uint8(denoised_im), vmin=0, vmax=255)
        axs[2].set_title('Denoised, PSNR = {:.2f}'.format(denoised_psnr))
        plt.show()

    return
   

# Description:
# Parsing command line
# 
# Outputs:
# opt - options
def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_folder', type=str, default='./data_set/cbsd68/', help='path to the input folder')
    parser.add_argument('--im_name', type=str, default='119082.png', help='image name')
    parser.add_argument('--out_folder', type=str, default='./output/images/demo/', help='path to the output folder')
    parser.add_argument('--sigma', type=int, default=25, help='noise sigma')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu_usage', type=int, default=0, help='0 - use CPU, 1 - use GPU')
    parser.add_argument('--plot', action='store_true', help='plot the processed image')
    parser.add_argument('--save', action='store_true', help='save the denoised image')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    process_image_func()
