import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import write_video
import scipy.misc
from scipy import signal
from scipy import ndimage
from pil_video import make_video

from torchvision.utils import save_image

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    assert is_sequence(inputs)
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        #img = tensor_to_image(tensor.cpu() )
        img = image_tensor(tensor)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        images.append(img.numpy())
    imageio.mimsave(filename, images)

def pred(validate_seq, validate_cond, modules, args, device):
    modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
    modules["posterior"].hidden = modules["posterior"].init_hidden()

    x = validate_seq
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]

    for i in range(1, args.n_past + args.n_future): # 1 ~ 11 
        h = modules["encoder"](x_in)
        h_target = modules["encoder"](x[i])[0].detach()

        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()

        if i < args.n_past: # n < 2 
            z_t, _, _ = modules['posterior'](h_target)
            modules["frame_predictor"](torch.cat([h, z_t, validate_cond[i-1]], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            z_t = torch.randn_like(z_t).to(device, dtype = torch.float32)
            h_pred = modules["frame_predictor"](torch.cat([h, z_t, validate_cond[i-1]], 1)).detach()
            x_in = modules["decoder"]([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    return posterior_gen

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(images, filename)

def plot_pred(validate_seq, validate_cond, modules, epoch, args, device):
    modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
    modules["posterior"].hidden = modules["posterior"].init_hidden()
    

    x = validate_seq
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]

    for i in range(1, args.n_past + args.n_future): # 1 ~ 11 
        h = modules["encoder"](x_in)
        h_target = modules["encoder"](x[i])[0].detach()

        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()

        if i < args.n_past: # n < 2 
            z_t, _, _ = modules['posterior'](h_target)
            modules["frame_predictor"](torch.cat([h, z_t, validate_cond[i-1]], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            z_t = torch.randn_like(z_t).to(device, dtype = torch.float32)
            h_pred = modules["frame_predictor"](torch.cat([h, z_t, validate_cond[i-1]], 1)).detach()
            x_in = modules["decoder"]([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    to_plot = []
    nrow = min(args.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(args.n_past+args.n_future):
            row.append(posterior_gen[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/pred_%d.png' % (args.log_dir, epoch) 
    save_tensors_image(fname, to_plot)

def normalize_data(sequence):
    sequence.transpose_(0, 1)
    return sequence

    
def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def plot_test(validate_seq, validate_cond, modules, epoch, args, device):
    
    to_image = transforms.ToPILImage()
    
    modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
    modules["posterior"].hidden = modules["posterior"].init_hidden()
    

    x = validate_seq
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]

    for i in range(1, args.n_past + args.n_future): # 1 ~ 11 
        h = modules["encoder"](x_in)
        h_target = modules["encoder"](x[i])[0].detach()

        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()

        if i < args.n_past: # n < 2 
            z_t, _, _ = modules['posterior'](h_target)
            modules["frame_predictor"](torch.cat([h, z_t, validate_cond[i-1]], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            z_t = torch.randn_like(z_t).to(device, dtype = torch.float32)
            h_pred = modules["frame_predictor"](torch.cat([h, z_t, validate_cond[i-1]], 1)).detach()
            x_in = modules["decoder"]([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    to_plot = []
    nrow = 1 
    gif =[]
    for i in range(nrow):
        row = []
        for t in range(args.n_past+args.n_future):
            row.append(posterior_gen[t][i]) 
            gif.append(to_image(posterior_gen[t][i]))
            
        to_plot.append(row)
    fname = '%s/gen/test_%d.png' % (args.log_dir, epoch) 
    
    gif[0].save("%s/genout.gif"% (args.log_dir) , save_all=True,optimize=False, append_images=gif[1:], loop=0, duration = 1)

    save_tensors_image(fname, to_plot)
    
    