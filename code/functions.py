import time
import cv2
import numpy as np
import os
import gc

from auxiliary_functions import *
from modules import *


# Description:
# This function filteres an input noisy image
#
# Inputs:
# noisy_im - noisy image
# sigma - noise sigma {15, 25, 50}
# gpu_usage - GPU usage: 
#             0 - use CPU, 
#             1 - use GPU for nearest neighbor search,
#             2 - use GPU for whole processing (requires more GPU memory)
#
# Outputs:
# denoised_im - denoised image
# denoising_time - denoising time 
def denoise_image(noisy_im, sigma, gpu_usage=0):
    s_cnn = ImCnn()
    
    if gpu_usage == 2:
        if torch.cuda.is_available():
            noisy_im = noisy_im.cuda()
            s_cnn.cuda()
        else:
            warnings.warn("CUDA isn't supported")

    state_file_name = './models/s_cnn_images/model_state_sig{}.pt'.format(sigma)
    assert os.path.isfile(state_file_name)
    print("=> loading model state '{}'".format(state_file_name))
    model_state = torch.load(state_file_name)
    s_cnn.load_state_dict(model_state['state_dict'])
    s_cnn.eval()

    with torch.no_grad():
        stime = time.time()
        im_n = reflection_pad_3d(noisy_im, (0, 6, 6))
        im_n = nn_func.pad(im_n, (37, 37, 37, 37, 0, 0), mode='constant', value=-1)
        denoised_im = s_cnn(im_n, gpu_usage).clamp(0, 1)
        etime = time.time()
        denoising_time = etime - stime
        denoised_im = denoised_im.cpu()

    return denoised_im, denoising_time


# Description:
# This function filteres an input noisy set
#
# Inputs:
# noisy_list - list of noisy images
# im_name_list - list of image names
# sigma - noise sigma {15, 25, 50}
# gpu_usage - GPU usage: 
#             0 - use CPU, 
#             1 - use GPU for nearest neighbor search,
#             2 - use GPU for whole processing (requires more GPU memory)
# silent - boolean flag: False - print "done" every image
#                         True - don't print "done" every image
#
# Outputs:
# denoised_set - list of denoised images
# denoising_time_list - list of denoising times
def denoise_image_list(noisy_list, im_name_list, sigma, gpu_usage, silent):
    s_cnn = ImCnn()

    if gpu_usage == 2: 
        if torch.cuda.is_available():
            s_cnn.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(s_cnn.parameters()).device

    state_file_name = './models/s_cnn_images/model_state_sig{}.pt'.format(sigma)
    assert os.path.isfile(state_file_name)
    print("=> loading model state '{}'".format(state_file_name))
    model_state = torch.load(state_file_name)
    s_cnn.load_state_dict(model_state['state_dict'])
    s_cnn.eval()

    with torch.no_grad():
        denoised_set = list()
        denoising_time_list = list()
        for i in range(len(noisy_list)):
            im_n = noisy_list[i].to(device)
            stime = time.time()
            im_n = reflection_pad_3d(im_n, (0, 6, 6))
            im_n = nn_func.pad(im_n, (37, 37, 37, 37, 0, 0), mode='constant', value=-1)
            denoised_im = s_cnn(im_n, gpu_usage).clamp(0, 1)
            etime = time.time()
            denoising_time = etime - stime
            denoised_im = denoised_im.cpu()
            denoised_set.append(denoised_im)
            denoising_time_list.append(denoising_time)

            if not silent:
                print('{}/{}: image {} done, denoising time: {:.4f}'.\
                    format(i + 1, len(noisy_list), os.path.splitext(im_name_list[i])[0].upper(), denoising_time))
                sys.stdout.flush()

    return denoised_set, denoising_time_list


# Description:
# This function filteres an input noisy video sequence
#
# Inputs:
# noisy_vid - noisy video sequence
# vid_name - video name
# clipped_noise - type of noise: 0 - AWGN, 1 - clipped Gaussian noise
# sigma - noise sigma {10, 20, 30, 40, 50}
# gpu_usage - GPU usage: 
#             0 - use CPU, 
#             1 - use GPU for nearest neighbor search,
#             2 - use GPU for whole processing (requires large GPU memory)
# silent - boolean flag: False - print "done" every frame
#                        True - don't print "done" every frame
#
# Outputs:
# denoised_vid - denoised video sequence
# denoising_time - denoising time 
def denoise_video_sequence(noisy_vid, vid_name, sigma, \
    clipped_noise=False, gpu_usage=0, silent=False):
    s_cnn = VidCnn()

    if gpu_usage == 2: 
        if torch.cuda.is_available():
            s_cnn.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(s_cnn.parameters()).device

    clip_str = 'clip_' if clipped_noise else ''
    state_file_name_s = './models/s_cnn_video/model_state_{}sig{}.pt'.format(clip_str, sigma)
    assert os.path.isfile(state_file_name_s)
    print("=> loading model state '{}'".format(state_file_name_s))
    model_state_s = torch.load(state_file_name_s)
    s_cnn.load_state_dict(model_state_s['state_dict'])
    s_cnn.eval()

    reflect_pad = (0, 10, 10)
    const_pad = (37, 37, 37, 37, 3, 3)
    with torch.no_grad():
        denoised_vid_s = torch.zeros_like(noisy_vid)
        noisy_vid_pad = reflection_pad_3d(noisy_vid, reflect_pad)
        noisy_vid_pad = nn_func.pad(noisy_vid_pad, const_pad, mode='constant', value=-1)

        denoising_time_s_list = list()
        for t_s in range(noisy_vid_pad.shape[-3] - 6):
            sliding_window = noisy_vid_pad[..., t_s:(t_s + 7), :, :].to(device)
            stime = time.time()
            denoised_frame_s = s_cnn(sliding_window, gpu_usage).clamp(min=0, max=1)
            etime = time.time()
            denoised_vid_s[..., t_s, :, :] = denoised_frame_s.cpu()
            denoising_time_s = etime - stime
            denoising_time_s_list.append(denoising_time_s)

            if not silent:
                print('S-CNN: frame {}/{} of sequence {} done, denoising time: {:.4f}'.\
                    format(t_s + 1, denoised_vid_s.shape[-3], vid_name.upper(), denoising_time_s))
                sys.stdout.flush()

    gc.collect()
    torch.cuda.empty_cache()

    t_cnn = TfNet()

    if gpu_usage > 0: 
        if torch.cuda.is_available():
            t_cnn.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(t_cnn.parameters()).device

    state_file_name_t = './models/t_cnn/model_state_{}sig{}.pt'.format(clip_str, sigma)
    assert os.path.isfile(state_file_name_t)
    print("=> loading model state '{}'".format(state_file_name_t))
    model_state_t = torch.load(state_file_name_t)
    t_cnn.load_state_dict(model_state_t['state_dict'])
    t_cnn.eval()

    with torch.no_grad():
        denoised_vid_t = torch.zeros_like(noisy_vid)
        cat_vid = torch.cat((noisy_vid, denoised_vid_s), dim=1)
        cat_vid = reflection_pad_t_3d(cat_vid, 3).to(device)

        denoising_time_t_list = list()
        for t_s in range(cat_vid.shape[2] - 6):
            sliding_window = cat_vid[:, :, t_s:(t_s + 7), ...].to(device)
            stime = time.time()
            denoised_frame_t = t_cnn(sliding_window).clamp(min=0, max=1)
            etime = time.time()
            denoised_vid_t[..., t_s, :, :] = denoised_frame_t.cpu()
            denoising_time_t = etime - stime
            denoising_time_t_list.append(denoising_time_t)

        if not silent:
            print('T-CNN: sequence {} done, denoising time: {:.4f}'.\
                format(vid_name.upper(), np.array(denoising_time_t_list).sum()))
            sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()

    denoising_time = np.array(denoising_time_s_list).sum() + \
        np.array(denoising_time_t_list).sum()

    return denoised_vid_t, denoised_vid_s, denoising_time


# Description:
# Reproducing denoising video sequence with bug during adding noise
#
# Inputs:
# clean_vid - clean video sequence
# vid_name - video name
# clipped_noise - tpe of noise: 0 - AWGN, 1 - clipped Gaussian noise
# sigma - noise sigma {10, 20, 30, 40, 50}
# gpu_usage - GPU usage: 
#             0 - use CPU, 
#             1 - use GPU for nearest neighbor search,
#             2 - use GPU for whole processing (requires large GPU memory)
# silent - boolean flag: False - print "done" every frame
#                        True - don't print "done" every frame
#
# Outputs:
# denoised_vid - denoised video sequence
# denoising_time - denoising time 
def denoise_video_sequence_with_bug(clean_vid, vid_name, sigma, \
    clipped_noise=False, gpu_usage=0, silent=False):
    s_cnn = VidCnn()

    if gpu_usage == 2: 
        if torch.cuda.is_available():
            s_cnn.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(s_cnn.parameters()).device

    clip_str = 'clip_' if clipped_noise else ''
    state_file_name_s = './models/s_cnn_video/model_state_{}sig{}.pt'.format(clip_str, sigma)
    assert os.path.isfile(state_file_name_s)
    print("=> loading model state '{}'".format(state_file_name_s))
    model_state_s = torch.load(state_file_name_s)
    s_cnn.load_state_dict(model_state_s['state_dict'])
    s_cnn.eval()

    reflect_pad = (0, 10, 10)
    const_pad_pixels = (37, 37, 37, 37, 0, 0)
    const_pad_frames = (0, 0, 0, 0, 3, 3)
    with torch.no_grad():
        denoised_vid_s = torch.zeros_like(clean_vid)
        noisy_vid = torch.zeros_like(clean_vid)
        clean_vid_frames_pad = nn_func.pad(clean_vid, const_pad_frames, mode='constant', value=-1)

        denoising_time_s_list = list()
        for t_s in range(clean_vid_frames_pad.shape[-3] - 6):
            clean_sliding_window = clean_vid_frames_pad[..., t_s:(t_s + 7), :, :].to(device)
            noisy_sliding_window = clean_sliding_window + (sigma / 255) * torch.randn_like(clean_sliding_window)
            if clipped_noise:
                noisy_sliding_window = torch.clamp(noisy_sliding_window, min=0, max=1)
            noisy_vid[..., t_s, :, :] = noisy_sliding_window[..., 3, :, :]
            noisy_sliding_window_pad = reflection_pad_3d(noisy_sliding_window, reflect_pad)
            noisy_sliding_window_pad = nn_func.pad(noisy_sliding_window_pad, const_pad_pixels, mode='constant', value=-1)
            stime = time.time()
            denoised_frame_s = s_cnn(noisy_sliding_window_pad, gpu_usage).clamp(min=0, max=1)
            etime = time.time()
            denoised_vid_s[..., t_s, :, :] = denoised_frame_s.cpu()
            denoising_time_s = etime - stime
            denoising_time_s_list.append(denoising_time_s)

            if not silent:
                print('S-CNN: frame {}/{} of sequence {} done, denoising time: {:.4f}'.\
                    format(t_s + 1, denoised_vid_s.shape[-3], vid_name.upper(), denoising_time_s))
                sys.stdout.flush()

    gc.collect()
    torch.cuda.empty_cache()

    t_cnn = TfNet()

    if gpu_usage > 0: 
        if torch.cuda.is_available():
            t_cnn.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(t_cnn.parameters()).device

    state_file_name_t = './models/t_cnn_bug/model_state_{}sig{}.pt'.format(clip_str, sigma)
    assert os.path.isfile(state_file_name_t)
    print("=> loading model state '{}'".format(state_file_name_t))
    model_state_t = torch.load(state_file_name_t)
    t_cnn.load_state_dict(model_state_t['state_dict'])
    t_cnn.eval()

    with torch.no_grad():
        denoised_vid_t = torch.zeros_like(noisy_vid)
        cat_vid = torch.cat((noisy_vid, denoised_vid_s), dim=1)
        cat_vid = reflection_pad_t_3d(cat_vid, 3).to(device)

        denoising_time_t_list = list()
        for t_s in range(cat_vid.shape[2] - 6):
            sliding_window = cat_vid[:, :, t_s:(t_s + 7), ...].to(device)
            stime = time.time()
            denoised_frame_t = t_cnn(sliding_window).clamp(min=0, max=1)
            etime = time.time()
            denoised_vid_t[..., t_s, :, :] = denoised_frame_t.cpu()
            denoising_time_t = etime - stime
            denoising_time_t_list.append(denoising_time_t)

        if not silent:
            print('T-CNN: sequence {} done, denoising time: {:.4f}'.\
                format(vid_name.upper(), np.array(denoising_time_t_list).sum()))
            sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()

    denoising_time = np.array(denoising_time_s_list).sum() + \
        np.array(denoising_time_t_list).sum()

    return denoised_vid_t, denoised_vid_s, denoising_time


# Description:
# This function reads an image from the memory
# 
# Inputs:
# file_name - path to the image
#
# Outputs:
# im - torch tensor of size (1 x 3 x 1 x image_height x image_width)
#      that contains the image
def read_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = (np.transpose(np.atleast_3d(im), (2, 0, 1)) / 255).astype(np.float32)
    im = torch.from_numpy(im)
    im = im.unsqueeze(1).unsqueeze(0)

    return im


# Description:
# This function saves an image to required location
# 
# Inputs:
# image_to_save - torch (CPU) tensor of size (3 x image_height x image_width) 
#                 that contains the image
# folder_name - path to the folder
# image_name - image name
def save_image(image_to_save, folder_name, image_name):
    image_to_save = float_tensor_to_np_uint8(image_to_save)
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(folder_name + image_name, image_to_save)

    return


# Description:
# Saving video sequence as AVI file
# 
# Inputs:
# sequence_to_save - torch (CPU) tensor of size (3 x number_of_frames X frame_height x frame_width) 
#                    that contains the video sequence
# folder_name - path to the folder
# vid_name - video name
def save_video_avi(sequence_to_save, folder_name, vid_name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 24
    t, h, w = sequence_to_save.shape[1:]
    f_name_avi = folder_name + vid_name + '.avi'
    vid_writer = cv2.VideoWriter(f_name_avi, fourcc, fps, (w, h))
    for i in range(t):
        frame_to_save = float_tensor_to_np_uint8(sequence_to_save[:, i, ...])
        frame_to_save = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
        vid_writer.write(frame_to_save)
    vid_writer.release()

    return


# Description:
# Converting float torch tensor to uint8 numpy array
# 
# Inputs:
# im_in - torch (CPU) float tensor of size (3 x image_height x image_width) 
#         that contains the image
# 
# Outputs:
# im_out - numpy uint8 array of size (image_height x image_width x 3)
def float_tensor_to_np_uint8(im_in):
    im_out = np.round(im_in.clamp(0, 1).clone().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return im_out


# Description:
# Read image set from a folder
# 
# Inputs:
# folder_name - path to the folder
#
# Outputs:
# im_list - list of torch tensors of size (1 x 3 x 1 x image_height x image_width)
#           that contains the image
# im_names - names of the read images
def read_image_set(folder_name):
    im_name_list = sorted(os.listdir(folder_name))

    im_list = list()    
    for i in range(len(im_name_list)):
        file_name = folder_name + im_name_list[i]
        im = cv2.imread(file_name, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = (np.transpose(np.atleast_3d(im), (2, 0, 1)) / 255).astype(np.float32)
        im = torch.from_numpy(im)
        im = im.unsqueeze(1).unsqueeze(0)
        im_list.append(im)

    return im_list, im_name_list


# Description:
# Read video sequence from a folder
# 
# Inputs:
# folder_name - path to the folder
# max_frame_num - maximum number of frames to read
#
# Outputs:
# vid - torch tensor of size (1 x 3 x number_of_frames x video_height x video_width)
#       that contains the video sequence
def read_video_sequence(folder_name, max_frame_num, file_ext):
    frame_name = folder_name + '{:05}.{}'.format(0, file_ext)
    frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = (np.transpose(np.atleast_3d(frame), (2, 0, 1)) / 255).astype(np.float32)

    frame_h, frame_v = frame.shape[1:3]
    frame_num = min(len(os.listdir(folder_name)), max_frame_num)
    vid = torch.full((3, frame_num, frame_h, frame_v), float('nan'))
    vid[:, 0, :, :] = torch.from_numpy(frame)

    for i in range(1, frame_num):
        frame_name = folder_name + '{:05}.{}'.format(i, file_ext)
        frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (np.transpose(np.atleast_3d(frame), (2, 0, 1)) / 255).astype(np.float32)
        vid[:, i, :, :] = torch.from_numpy(frame)

    vid = vid.unsqueeze(0)

    return vid
