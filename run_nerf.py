import os, sys
import cupy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import cmath,math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from run_nerf_helpers import *
import os,gc
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def get_3d_vector(azimuth_rad, elevation_rad):
    x = np.cos(azimuth_rad) * np.cos(elevation_rad)
    y = np.sin(azimuth_rad) * np.cos(elevation_rad)
    z = np.sin(elevation_rad)
    z=np.tile(z, (1,360, 1, 1))
    #vector = np.array([x, y, z])
    #print("get_3d",x.shape,y.shape,z.shape)
    vector=np.concatenate((x, y,z), axis=-1)
    return vector

def BRDF(normal,in_normal,out_normal,miu1,ixron1,miu2,ixron2,albedo,roughness):
    #print("miu",miu1.shape)
    #print("ixron1",ixron1.shape)
    #print("in_normal",in_normal.shape,out_normal.shape,normal.shape)
    #return 1+1j
    #print("normal",normal.shape)
    in_normal=torch.tensor(in_normal).float()
    #print("in_normal",in_normal.shape)
    if len(in_normal.size())==2:
        in_normal=torch.unsqueeze(in_normal,dim=1)
    out_normal=torch.tensor(out_normal).float()
    #print("normal",normal.shape,out_normal.shape)
    reflect_coefficient1=torch.sqrt(miu1/ixron1)
    reflect_coefficient2=torch.sqrt(miu2/ixron2)
    #print(torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1))
    #print((torch.sqrt(torch.sum(out_normal*out_normal,dim=-1))*torch.sqrt(torch.sum(normal*normal,dim=-1))))
    #print(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1).shape)
    #print((in_normal*in_normal).shape)
    #print((normal*normal).shape)
    cos_in=torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1)/(torch.sqrt(torch.sum(in_normal * in_normal, dim=-1))*torch.sqrt(torch.sum(normal*normal,dim=-1)))
    cos_out=torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1)/(torch.sqrt(torch.sum(out_normal*out_normal,dim=-1))*torch.sqrt(torch.sum(torch.squeeze(normal*normal,dim=-2),dim=-1)))
    #normal=torch.unsqueeze(normal,axis=-1)
    #in_normal=torch.unsqueeze(in_normal,axis=-1)
    #out_normal=torch.unsqueeze(out_normal,axis=-1)
    #print("reflect_coefficient",reflect_coefficient2.shape,cos_in.shape,cos_out.shape)
    #print(torch.unsqueeze(cos_in,dim=2)*reflect_coefficient1)
    #print((torch.unsqueeze(cos_in,dim=2)*reflect_coefficient1).shape,(reflect_coefficient2*torch.reshape(cos_out,(cos_out.shape[0],1,1))).shape)
    F_S=(torch.unsqueeze(cos_in,dim=2)*reflect_coefficient1-reflect_coefficient2*torch.reshape(cos_out,(cos_out.shape[0],1,1)))/(torch.unsqueeze(cos_in,dim=2)*reflect_coefficient1+reflect_coefficient2*torch.reshape(cos_out,(cos_out.shape[0],1,1)))
    #print(F_S.shape)
    #print(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1).shape)
    #print((albedo/math.pi).shape)
    Lambert=torch.reshape(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1),(out_normal.shape[0],1,1))*(1-F_S)*(albedo/math.pi)
    h=(in_normal+torch.reshape(out_normal,(out_normal.shape[0],1,-1)))/2
    #print((roughness**4-1).shape)
    #print(h*normal)
    #print((torch.sum(h*normal,dim=-1)**2).shape)
    #print((roughness**4-1))
    #print((math.pi*(torch.unsqueeze(torch.sum(h*normal,dim=-1),dim=-1)**2*(roughness**4-1)+1)**2).shape)
    D=roughness**4/(math.pi*(torch.unsqueeze(torch.sum(h*normal,dim=-1),dim=-1)**2*(roughness**4-1)+1)**2)
    k=roughness**4/2
    #print(torch.matmul(in_normal,normal))
    #print((torch.outer(torch.matmul(in_normal,normal),(1-k))+k).shape)
    #print((torch.matmul(in_normal,normal)*out_normal.dot(normal)).unsqueeze(1).repeat(1, 256).shape)
    #print(((torch.outer(torch.matmul(in_normal,normal),(1-k))+k)*(out_normal.dot(normal)*(1-k)+k)).shape)
    #print(torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1).shape)
    #print(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1).shape)
    #print(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1).shape)
    #print(k.shape)
    #print((torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1)*torch.unsqueeze(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1),dim=-1)).shape)
    #print(((torch.unsqueeze(torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1),dim=-1)*(1-k)+k)*(torch.reshape(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1),(-1,1,1))*(1-k)+k)).shape)
    G=((torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1)*torch.unsqueeze(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1),dim=-1)).unsqueeze(-1).repeat(1,1, 256))/((torch.unsqueeze(torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1),dim=-1)*(1-k)+k)*(torch.reshape(torch.sum(out_normal*torch.squeeze(normal,dim=-2),dim=-1),(-1,1,1))*(1-k)+k))
    #print(D.shape,F_S.shape,G.shape,torch.matmul(in_normal,normal).shape)
    cook=D*F_S*G/(4*torch.sum(in_normal*normal.repeat(1,in_normal.shape[1],1),dim=-1).unsqueeze(-1).repeat(1,1,256))
    #print(cook.shape)
    #print(cook+Lambert)
    function_values=cook+Lambert
    where_are_nan = torch.isnan(function_values)
    where_are_inf = torch.isinf(function_values)
    function_values[where_are_nan] = 0
    function_values[where_are_inf] = 0
    #print(function_values.shape)
    return function_values

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        #print(inputs)
        #inputs=torch.tensor(inputs)
        inputs=inputs.clone().detach().requires_grad_(True)
        #print(fn(inputs[0:1])[0])
        return torch.cat([(fn(inputs[i:i+chunk])[0]) for i in range(0, inputs.shape[0], chunk)], 0),torch.cat([(fn(inputs[i:i+chunk])[1]) for i in range(0, inputs.shape[0], chunk)], 0),torch.cat([(fn(inputs[i:i+chunk])[2]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    
    '''if viewdirs is not None:
        viewdirs=torch.tensor(viewdirs)
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        #print(input_dirs_flat)
        #print(embeddirs_fn)
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)'''

    outputs_flat = batchify(fn, netchunk)(embedded)
    #print(outputs_flat)
    outputs1 = torch.reshape(outputs_flat[0], list(inputs.shape[:-1]) + [outputs_flat[0].shape[-1]])
    outputs2 = torch.reshape(outputs_flat[1], list(inputs.shape[:-1]) + [outputs_flat[1].shape[-1]])
    outputs3 = torch.reshape(outputs_flat[2], list(inputs.shape[:-1]) + [outputs_flat[2].shape[-1]])
    return outputs1,outputs2,outputs3


def batchify_rays(view_i,Rx,rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(view_i,Rx,rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(batch_size,view_i,Rx,H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o = [[1,2,3]] # [N_rays, 3] each
        rays_d=viewdirs = np.random.rand(1,3)

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = torch.tensor(rays_d)
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o=torch.Tensor(rays_o)
    rays_d=torch.from_numpy(rays_d)
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o.to('cuda'), rays_d.to('cuda'), near.to('cuda'), far.to('cuda')], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(view_i,Rx,rays, chunk, **kwargs)
    #print(all_ret)
    list_add=[batch_size,]
    for k in all_ret:
        #print(list(tuple(batch_size,)))
        #print(list(all_ret[k].shape[1:]))
        k_sh = list_add + list(all_ret[k].shape[1:])
        #print("k_sh",k_sh)
        #print("all_ret[k]",all_ret[k].shape)
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(2,H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    output_ch2=6
    skips = [4]
    model = NeRF(D=8, W=256,
                 input_ch=input_ch, output_ch1=output_ch, output_ch2=output_ch2, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch1=output_ch, output_ch2=output_ch2, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def shade(network_query_fn,A,dists,raw,raw2,raw3,viewdirs,pts,Tx,Rx,network_fn,random_p):
    #print("dists",dists.shape)
    #print("raw",raw.shape,raw2.shape,raw3.shape)
    #print("shape",viewdirs.shape,pts.shape)
    #print("viewdirs",viewdirs.shape)
    if dists.dim()==3:
        dists=torch.squeeze(dists,axis=-1)
    f_1=np.linspace(77, 81, num=256)
    f=np.reshape(f_1, (1, 1, 1, 256))
    #c=3*pow(10,8)
    c=300
    PRR=random_p
    #print(raw.shape)
    if random.random()>PRR:
        return torch.zeros((raw.shape[0],2,8,256))
    #print("raw_dists",dists.shape)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    #print("raw_alpha",alpha.shape)
    T=torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    #print(pts.shape)
    #print("raw_T",T.shape)
    new_outdirs=np.zeros((pts.shape[0],3))
    #brdfs=np.zeros((pts.shape[0],pts.shape[1],1))+1j*np.zeros((pts.shape[0],pts.shape[1],1))
    #brdfs=torch.empty(pts.shape[0], 256,dtype=torch.complex64)
    possibility=np.zeros((pts.shape[0],1))
    #print(pts.shape)
    #for pts0 in range(pts.shape[0]):
        #print(pts0)
        #for pts1 in range(pts.shape[1]):
            #print(pts1)
    pts1=0
    elevation=np.empty((0,1,90,1))
    for pts0 in range(pts.shape[0]):
        if raw[pts0][pts1][2]>0:
            elevation1=np.linspace(0, math.pi/2, 90)
        else:
            elevation1=np.linspace(-math.pi/2, 0, 90)
        elevation1=elevation1.reshape((1,1,90,1))
        elevation=np.concatenate((elevation,elevation1),axis=0)
    azimuth=np.linspace(0,2*math.pi,360)
    #result_array = []
    azimuth=azimuth.reshape((1,360,1,1))
    azimuth=np.tile(azimuth,(pts.shape[0],1,1,1))
    #for azi in range(len(azimuth)):
        #for ele in range(len(elevation)):
    result_array=get_3d_vector(azimuth,elevation)
    #print("result_array shape",result_array.shape)
    result_array=np.reshape(result_array, (pts.shape[0],result_array.shape[1]*result_array.shape[2], 3))
    #print(vector)
    #result_array.append(vector)
    #result_array=np.array(result_array)
    #x_pdf=np.linspace(0, azimuth.shape[1]*elevation.shape[2]-1, azimuth.shape[1]*elevation.shape[2])
    #x_pdf=np.tile(x_pdf,(pts.shape[0],1))
    #function_values=[]
    a5,c5,d5,e5,f5,g5,h5,t5=raw[:,:,0:-2].detach(),viewdirs,raw2[:,:,0].detach(),raw2[:,:,1].detach(),raw2[:,:,2].detach(),raw2[:,:,3].detach(),raw2[:,:,4].detach(),raw2[:,:,5].detach()
    #a,c,d,e,f,g,h,t=raw[pts0][pts1][0:-2],viewdirs[pts0],raw2[pts0][pts1][0],raw2[pts0][pts1][1],raw2[pts0][pts1][2],raw2[pts0][pts1][3],raw2[pts0][pts1][4],raw2[pts0][pts1][5]
    #for i in range(len(result_array)):
        #function_values.append(np.array(torch.sum(torch.real(BRDF(a5,result_array[i],c5,d5,e5,f5,g5,h5,t5)),dim=-1).to('cpu')))
    #print("array",result_array.shape)
    function_values=np.array(torch.sum(torch.real(BRDF(a5,result_array,c5,d5,e5,f5,g5,h5,t5)),dim=-1).to('cpu'))
    #print("function_values",function_values.shape)
    #function_values=np.array(function_values)
    #function_values=np.random.random((len(result_array)))
    #print(function_values)
    where_are_nan = np.isnan(function_values)
    where_are_inf = np.isinf(function_values)
    function_values[where_are_nan] = 0
    function_values[where_are_inf] = 0
    #print(function_values)
    #print(x_pdf.shape)
    for pts0 in range(pts.shape[0]):
        x_pdf=np.linspace(0, azimuth.shape[1]*elevation.shape[2]-1, azimuth.shape[1]*elevation.shape[2])
        pdf=gaussian_kde(function_values[pts0]).evaluate(x_pdf)
    #print(pdf.shape)
    #print(pdf.ravel())
        p=pdf.ravel()/pdf.ravel().sum()
        where_are_nan = np.isnan(p)
        where_are_inf = np.isinf(p)
        p[where_are_nan] = 0
        p[where_are_inf] = 0
        #print(p.shape)
        index = np.random.choice(x_pdf, p=p)
        #print(int(index))
        possibility[pts0]=pdf[int(index)]
        new_outdir=result_array[pts0][int(index)]
        new_outdirs[pts0]=new_outdir
    
    #print("outdir",new_outdir.shape,new_outdirs.shape)
    brdf=BRDF(a5,new_outdirs,c5,d5,e5,f5,g5,h5,t5)
    #print("brdf_shape",brdf.shape)
    brdfs=torch.squeeze(brdf,1)
    brdfs=brdfs.to('cuda')
    tensor1 = pts.detach().to("cpu").numpy()
    tensor2 = np.array(Rx)
    #print("pts",tensor1.shape)
    #print("Rx",tensor2.shape)
    tensor1_reshaped = tensor1[:, :, np.newaxis, :] 
    #tensor2_reshaped = tensor2[:, np.newaxis,:, :]      #now for one Rx, tensor2[:, np.newaxis, :, :]
    output_tensor = tensor1_reshaped * tensor2
    #print("output_tensor",output_tensor.shape)
    #output_tensor = np.squeeze(output_tensor)
    output_tensor = np.sum(output_tensor, axis=-1, keepdims=True)
    #print(output_tensor.shape)
    distance=output_tensor
    #distance=np.concatenate((distance, distance),axis=1)
    #print("distance",distance.shape)
    #a=2*math.pi*f/c*distance*2j
    w_d_para_old=2*math.pi/c*distance*2j
    #print("w_d_para_old",w_d_para_old.shape)
    #W_d_para=w_d_para_old[...,np.newaxis]
    #print("W_d_para",W_d_para.shape)
    #print("f",f.shape)
    W_d=2*np.exp(w_d_para_old*f)
    #print("W_d",W_d.shape)
    predict_depth=raw[...,-2:-1]
    #S_indirect=0
    S_direct=0
    point1 = pts
    point2 = torch.unsqueeze(torch.tensor(Tx),dim=-2)
    vector = point2 - point1
    #print("pts",pts.shape,point2.shape)
    #print(predict_depth.shape)
    #print(new_outdirs.shape)
    new_pts=pts+predict_depth*torch.unsqueeze(torch.tensor(new_outdirs).to('cuda'),1)
    #print("new_pts",new_pts.shape)
    #print("vector",vector.shape)
    #vector=torch.from_numpy(vector).to('cuda')
    new_pts1=torch.cat((new_pts,vector),2)
    raw,raw2,raw3 = network_query_fn(new_pts1, new_outdirs, network_fn) #network_query_fn(new_pts, new_outdirs, vector, network_fn)
    #print("raw",raw.shape,raw2.shape,raw3.shape)
    raw,raw2,raw3=raw[:,0:1,:],raw2[:,0:1,:],raw3[:,0:1,:]
    #print("raw",raw.shape)
    raw2 =torch.reshape(raw2,(*raw2.shape[:-1],6, 256))
    #print(T.shape,alpha.shape,brdfs.shape)
    T=torch.unsqueeze(T,axis=-1)
    alpha=torch.unsqueeze(alpha,axis=-1)
    #print(W_d[...,0].shape)
    #for rx_num in range(8):
    raw3=torch.unsqueeze(raw3,axis=-2)
    eq2=torch.unsqueeze(T**2*alpha,axis=-1)
    #eq2=T**2*alpha
    #print("T",T.shape)
    #print(alpha.shape)
    #print("eq2",eq2.shape)
    eq2=eq2*torch.from_numpy(W_d[...,:]).to('cuda')
    #print("eq2",eq2.shape)
    #print("raw3",raw3.shape)
    #print("brdfs",brdfs.shape)
    #eqq=raw3*eq2
    brdfs=brdfs.view(brdfs.shape[0],1,1,-1)
    eq1=raw3*eq2*brdfs
    #print("eq1",eq1.shape)
    #print("distance",distance.shape)
    S_direct+=eq1/torch.from_numpy(distance[...,:]).to('cuda')**2
    #print("s_direct",S_direct.shape)
    new_A=np.real(S_direct)
    #predict_depth=torch.squeeze(predict_depth,axis=-1)
    #print(brdfs.shape)
    #a=torch.rand(1, 2, 1)*torch.tensor(brdfs).to('cuda')*torch.exp(2*math.pi*f/c*predict_depth*2j)/torch.tensor(possibility).to('cuda')
    #print(a.shape)
    #shade(network_query_fn,new_A,predict_depth,raw,raw2,raw3,new_outdirs,new_pts,Tx,Rx,network_fn)
    f=torch.from_numpy(f).to('cuda')
    #print("f",f.shape)
    #print(predict_depth.shape)
    #print("possibility",possibility.shape)
    eq3=torch.exp(2*math.pi*f/c*torch.unsqueeze(predict_depth,dim=-1)*2j)/torch.from_numpy(possibility).to('cuda').view(-1,1,1,1)/PRR
    #print("eq3",eq3.shape)
    #print("brdfs",brdfs.shape)
    S_indirect=shade(network_query_fn,new_A,predict_depth,raw,raw2,raw3,new_outdirs,new_pts,Tx,Rx,network_fn,0.5)*eq3*brdfs
    #print("indirect",S_indirect.shape)
    return S_direct+S_indirect

def raw2outputs(network_query_fn,network_fn,A,Rx,raw, raw2,raw3,pts,Tx,viewdirs, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        raw2: Second prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    f=1
    c=3*pow(10,8)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    #print(alpha.shape)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    #A=torch.cat([torch.ones((alpha.shape[0], 1)), 1. + 1e-10], -1)
    T=torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T
    #print(raw.shape)
    #print(raw2.shape)
    #print(raw3.shape)
    #print(shade(network_query_fn,A,dists,raw,raw2,raw3,viewdirs,pts,Tx,Rx,network_fn))
    final_shade=0
    for sample_number in range(raw.shape[1]):
        final_shade+=torch.sum(shade(network_query_fn,A[:,sample_number:sample_number+1,:],dists[:,sample_number:sample_number+1],raw[:,sample_number:sample_number+1,:],raw2[:,sample_number:sample_number+1,:,:],raw3[:,sample_number:sample_number+1,:],viewdirs,pts[:,sample_number:sample_number+1,:],Tx,Rx,network_fn,1)/raw.shape[1],dim=1)
        #final_shade=torch.rand((2, 2, 8, 256))
    #rgb_map = torch.sum(shade(network_query_fn,A,dists,raw,raw2,raw3,viewdirs,pts,Tx,Rx,network_fn,1), -2)  # [N_rays]
    rgb_map=final_shade
    #print("rgb",rgb_map.shape)
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(view_i,Rx,ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    #A=1   # amplitude before
    N_samples=2
    Tx=[]
    for view_i_num in view_i:
        Tx.append([9.53*0.001*math.sin(360/1200*view_i_num/180*2*math.pi),9.53*0.001*math.cos(360/1200*view_i_num/180*2*math.pi),0])
    N_rays = len(view_i)
    #print(N_rays)
    rays_o = Tx # [N_rays, 3] each
    view_normal=[]
    for view_i_num in view_i:
        view_normal.append([1*math.cos(360/1200*view_i_num/180*2*math.pi),1*math.sin(360/1200*view_i_num/180*2*math.pi),0])
    rays_d=viewdirs =np.array(view_normal)
    #bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = 2*torch.ones((N_rays,1)),6*torch.ones((N_rays,1))

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    #print(rays_o)
    #print(rays_d)
    #print(z_vals)
    rays_o=torch.tensor(rays_o)
    rays_d=torch.tensor(rays_d)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    point1 = pts
    point2 =  torch.unsqueeze(torch.tensor(Tx),dim=-2)
    #print(point1.shape,point2.shape)
    vector = point2 - point1
#     raw = run_network(pts)
    #print("pts_ok",pts.shape)
    #print("vector_ok",vector.shape)
    new_pts=torch.cat((pts,vector),2)
    #print("new_pts_ok",new_pts.shape)
    #print("viewdirs",viewdirs.shape)
    raw,raw2,raw3 = network_query_fn(new_pts, viewdirs, network_fn)
    raw2 =torch.reshape(raw2,(*raw2.shape[:-1],6, 256))
    A=raw3
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(network_query_fn,network_fn,A,Rx,raw,raw2,raw3,pts,Tx,viewdirs, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw,raw2,raw3 = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(network_query_fn,network_fn,A,Rx,raw, raw2,raw3,z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.
    elif args.dataset_type == 'rf':
        images=torch.load('save_tensor.pt')
        i_train, i_val, i_test = np.arange(0, 1000, 1, int),np.arange(1000, 1100, 1, int),np.arange(1100, 1200, 1, int)

        near = 2.
        far = 6.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    #Rx=torch.random((1,2500,3))
    # Cast intrinsics to right types
    H, W, focal = 5,5,1
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    #render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    #poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 100 + 1
    print('Begin')
    #print('TRAIN views are', i_train)
    #print('TEST views are', i_test)
    #print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        #target_s=torch.rand((1,2))
        #torch.save(target_s, './final_result/save_tensor'+str(i)+".pt")
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        '''else:
            # Random from one image
            img_i = 0
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            #pose = poses[img_i, :3,:4]

            if N_rand is not None:
                #rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                #print(coords)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                #rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                #rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                #batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)'''
        
        #####  Core optimization loop  #####
        batch_size_new=10
        optimizer.zero_grad()
        loss=0
        for batch_iter in range(800//batch_size_new):
            view_i=list(range(batch_iter*batch_size_new,(batch_iter+1)*batch_size_new))
        #Rx=[[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]]
        #loss=0
        #view_i=list(range(0,1000))
            Rx=[]
            for view_i_num in view_i:
                Rx_view=[]
                for rx_num in range(8):
                    Rx_view.append([(-9.53+1.9*rx_num)*0.001*math.sin(360/1200*view_i_num/180*2*math.pi),(-9.53+1.9*rx_num)*0.001*math.cos(360/1200*view_i_num/180*2*math.pi),0])
                Rx.append([Rx_view])
            #print("Rx",Rx)
            batch_size=len(view_i)
            rgb, disp, acc, extras = render(batch_size,view_i,Rx,H, W, K, chunk=args.chunk,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)
            target_s=torch.empty((0,8,256))
            for view_num in view_i:
                target_s_single=torch.load('save_tensor.pt')[view_num:view_num+1].to('cuda')
                #target_s_single=torch.unsqueeze(target_s_single,0)
                #target_s_single=torch.unsqueeze(target_s_single,0)
                target_s=torch.cat((target_s, target_s_single), dim=0)
            #target_s=torch.unsqueeze(target_s,0)
            #target_s=torch.unsqueeze(target_s,0)
            #print("this is",target_s.shape,rgb.shape)
            img_loss = img2mse(rgb, target_s)
            #print("loss",img_loss)
            trans = extras['raw'][...,-1]
            loss += img_loss
            psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            global_step += 1
            view_i=list(range(801,1200))
            #Rx=[[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]]
            Rx=[]
            for view_i_num in view_i:
                Rx_view=[]
                for rx_num in range(8):
                    Rx_view.append([(-9.53+1.9*rx_num)*0.001*math.sin(360/1200*view_i_num/180*2*math.pi),(-9.53+1.9*rx_num)*0.001*math.cos(360/1200*view_i_num/180*2*math.pi),0])
                Rx.append([Rx_view])
            #print("Rx",Rx)
            batch_size=len(view_i)
            rgb, disp, acc, extras = render(batch_size,view_i,Rx,H, W, K, chunk=args.chunk,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)
            target_s=torch.empty((0,8,256))
            for view_num in view_i:
                target_s_single=torch.load('save_tensor.pt')[view_num:view_num+1].to('cuda')
                #target_s_single=torch.unsqueeze(target_s_single,0)
                #target_s_single=torch.unsqueeze(target_s_single,0)
                target_s=torch.cat((target_s, target_s_single), dim=0)
            #target_s=torch.unsqueeze(target_s,0)
            #target_s=torch.unsqueeze(target_s,0)
            torch.save(target_s, './final_result/save_tensor'+str(i)+".pt")
            #print("this is",target_s.shape,rgb.shape)
            img_loss = img2mse(rgb, target_s)
            #print("loss",img_loss)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)
            tqdm.write(f"[Test] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
