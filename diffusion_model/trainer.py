import torch
from torch import nn, einsum
from torchvision import utils
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import copy

from torch.utils import data
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import datetime
import time
from pathlib import Path
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



torch.cuda.empty_cache()

def cycle(dl):
    while True:
        for data in dl:
            yield data

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        #print("fp16 not set up")
        loss.backward(**kwargs)

    else:
        loss.backward(**kwargs)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def display(tensor):
    image_np = tensor.numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    print(image_np.shape)
    plt.axis('off')  # Hide axis
    plt.show()

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c = None, label_tensors=None, linear_tensors=None):    
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x, c], 1), t, y=label_tensors, l=linear_tensors))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False, label_tensors=None, linear_tensors=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised, label_tensors=label_tensors, linear_tensors=linear_tensors)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None, label_tensors=None, linear_tensors=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        # Looping through timesteps in reverse order
        for i in tqdm(reversed(range(self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition_tensors=condition_tensors, label_tensors=label_tensors, linear_tensors=linear_tensors)

        return img

    @torch.no_grad()
    def sample(self, batch_size = 16, conditions = None):
        condition_tensors = conditions[0]
        label_tensors = conditions[1]
        linear_tensors = conditions[2]
        condition_tensors = torch.zeros_like(condition_tensors).cuda()
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), condition_tensors=condition_tensors, label_tensors=label_tensors, linear_tensors=linear_tensors)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, condition_label = None, condition_tensors = None, linear_tensors=None, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start.float()))
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        if condition_tensors == None:
            condition_tensors = torch.zeros(b, c, h, w).cuda()  
        
        x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t, y=condition_label, l=linear_tensors)
        
        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, condition_label=None, condition_tensors=None, linear_tensors=None, *args, **kwargs):
        b, c, h, w = *x.shape,
        device = x.device
        img_size = self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors,condition_label=condition_label, linear_tensors=linear_tensors,*args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay = 0.995,
        train_batch_size = 32,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        save_and_sample_every = 1000,
        results_folder = 'results',
        results_iteration = '1'
    ):
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        
        self.save_and_sample_every = save_and_sample_every

        self.step_start_ema = step_start_ema

        self.gradient_accumulate_every = gradient_accumulate_every
        self.ds = dataset
        self.train_lr = train_lr
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.epoch_steps = len(dataset) // (self.batch_size*self.gradient_accumulate_every)

        self.dl = cycle(data.DataLoader(
            self.ds,
            batch_size=train_batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=4,  # Adjust based on your system
            persistent_workers=True
        ))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.fp16 = fp16
        if fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        results_parent_path =  os.path.join(path, '..', results_folder)

        '''i = 1
        self.results_path = os.path.join(results_parent_path, f'results-{i}')
        while os.path.exists(self.results_path):
            i += 1
            self.results_path = os.path.join(results_parent_path, f'results-{i}')'''

        self.results_path = os.path.join(results_parent_path, f'results-{results_iteration}')
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        location = str(os.path.join(self.results_path, f'sample-{milestone}.pt'))
        torch.save(data, location)

    def load(self, milestone):
        location = str(os.path.join(self.results_path, f'sample-{milestone}.pt'))
        data = torch.load(location)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        print("Training Started")

        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()
        epoch_time = time.time()

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                input_tensors = data['input'].cuda()
                input_label = data['label'].cuda()
                input_linear_cond = data['linear_condition'].cuda()
                
                self.opt.zero_grad()

                if self.fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.model(input_tensors, condition_label=input_label, linear_tensors=input_linear_cond)
                        loss = output.sum()/self.batch_size
                        self.scaler.scale(loss/self.gradient_accumulate_every).backward()
                else:
                    output = self.model(input_tensors, condition_label=input_label)
                    loss = output.sum()/self.batch_size
                    (loss / self.gradient_accumulate_every).backward()
                                
                print(f'{self.step}: {loss.item()}')
            
            if self.fp16:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(4, self.batch_size)
                
                all_images_list = list(map(
                    lambda n: self.ema_model.sample(
                        batch_size=n,
                        conditions = self.ds.sample_conditions(batch_size=n)
                    ),
                    batches
                ))

                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                
                print("Joining")
                location = str(os.path.join(self.results_path, f'sample-{milestone}.png'))
                utils.save_image(all_images, location, nrow = 2)
                self.save(milestone)

                print("SAVED")
            
            # Epoch Time Print
            if ((self.step % self.epoch_steps) == 0) and (self.step != 0):
                epoch_no = int(self.step / self.epoch_steps)
                
                current_time = time.time()
                epoch_time = (current_time-epoch_time)/60

                print(f'Epoch {epoch_no}: Time = {epoch_time} mins')

                epoch_time = current_time

            self.step += 1

        print("Training Completed")
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        print(execution_time)