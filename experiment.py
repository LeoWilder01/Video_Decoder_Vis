"""Decoder-only latent intervention experiments. Inputs must be diffusion-space CTHW latents."""
import argparse
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
from taehv import TAEHV

def load_decoder(device):
    # Construct on meta: no encoder weight storage is ever allocated.
    with torch.device('meta'):
        model = TAEHV(checkpoint_path=None, arch_name='taew2_1')
    del model.encoder
    model.to_empty(device=device)
    model.load_state_dict(load_file(str(ROOT/'weights/taew2_1_decoder.safetensors')))
    return model.eval().requires_grad_(False)

def selection(shape, mode, c, t, y, x, extent):
    C,T,H,W = shape
    if not all(0 <= i < n for i,n in zip((c,t,y,x), shape)):
        raise ValueError(f'Index {(c,t,y,x)} outside {tuple(shape)}')
    if mode == 'scalar':
        return (slice(c,c+1),slice(t,t+1),slice(y,y+1),slice(x,x+1))
    if mode == 'channel':
        return (slice(c,c+1),slice(0,T),slice(0,H),slice(0,W))
    dt,dy,dx = extent
    if min(extent) < 1 or t+dt>T:
        raise ValueError('Invalid time extent')
    if mode == 'time':
        return (slice(0,C),slice(t,t+dt),slice(0,H),slice(0,W))
    if y+dy>H or x+dx>W:
        raise ValueError('Region extends outside latent')
    return (slice(c,c+1),slice(t,t+dt),slice(y,y+dy),slice(x,x+dx))

def write_video(path, rgb, fps):
    frames = (rgb[0].permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--latent', type=Path, help='.npy or .safetensors, single CTHW tensor')
    source.add_argument('--synthetic', action='store_true', help='Plumbing test only; not semantic evidence')
    p.add_argument('--latent-space', choices=['diffusion'], help='Required acknowledgement for imported latents')
    p.add_argument('--key', default='latent')
    p.add_argument('--mode', choices=['scalar','region','channel','time'], default='scalar')
    p.add_argument('--channel',type=int,default=0)
    p.add_argument('--time',type=int,default=1)
    p.add_argument('--y',type=int,default=4)
    p.add_argument('--x',type=int,default=4)
    p.add_argument('--extent',type=int,nargs=3,default=[1,2,2],metavar=('DT','DY','DX'))
    p.add_argument('--deltas',type=float,nargs='+',default=[-1,0,1])
    p.add_argument('--units',choices=['absolute','channel_std'],default='absolute')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--fps',type=int,default=16,help='Use source video FPS for a real latent')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        p.error('Choose a new output directory to preserve previous experiments')
    if args.synthetic:
        z=torch.randn((16,3,8,8),generator=torch.Generator().manual_seed(42))
    else:
        if args.latent_space != 'diffusion':
            p.error('Confirm exported tensor is diffusion-space using --latent-space diffusion')
        z=(torch.from_numpy(np.load(args.latent,allow_pickle=False)) if args.latent.suffix=='.npy'
           else load_file(str(args.latent))[args.key])
    z=z.float().cpu()
    if z.ndim!=4 or z.shape[0]!=16 or min(z.shape)<1 or not torch.isfinite(z).all():
        raise ValueError('Expected a finite, nonempty [16,T,H,W] tensor')
    if not all(np.isfinite(args.deltas)) or args.fps<=0:
        raise ValueError('Deltas must be finite; FPS must be positive')
    sel=selection(z.shape,args.mode,args.channel,args.time,args.y,args.x,args.extent)
    model=load_decoder(args.device)
    def decode(value):
        start=time.perf_counter()
        # CTHW -> NTCHW. TAE expects diffusion-space values without further affine scaling.
        with torch.inference_mode():
            rgb=model.decode_video(value.permute(1,0,2,3).unsqueeze(0).to(args.device),
                                   parallel=False,show_progress_bar=False).cpu()
        return rgb,time.perf_counter()-start
    base,base_seconds=decode(z)
    assert tuple(base.shape)==(1,4*z.shape[1]-3,3,8*z.shape[2],8*z.shape[3])
    args.out.mkdir(parents=True)
    write_video(args.out/'baseline.mp4',base,args.fps)
    np.save(args.out/'baseline-latent.npy',z.numpy())
    report={'synthetic':args.synthetic,'semantic_evidence':False if args.synthetic else 'requires visual review',
            'source':str(args.latent) if args.latent else 'seed=42 Gaussian plumbing test',
            'space':'diffusion','latent_shape':list(z.shape),'rgb_shape_NTCHW':list(base.shape),
            'mode':args.mode,'selection_half_open':[[s.start,s.stop] for s in sel],
            'baseline_decode_seconds':base_seconds,'device':args.device,'fps':args.fps,'edits':[]}
    for index,delta in enumerate(args.deltas):
        edit=torch.zeros_like(z)
        if args.units=='channel_std':
            scale=z.std(dim=(1,2,3),correction=0).reshape(16,1,1,1).expand_as(z)
            edit[sel]=delta*scale[sel]
        else:
            edit[sel]=delta
        changed=z+edit
        rgb,seconds=decode(changed)
        diff=(rgb-base).abs()
        name=f'edit-{index:02d}'
        write_video(args.out/f'{name}.mp4',rgb,args.fps)
        np.save(args.out/f'{name}-latent-slice.npy',changed[args.channel,args.time].numpy())
        np.save(args.out/f'{name}-difference-per-pixel.npy',diff.mean(dim=2)[0].numpy())
        report['edits'].append({'name':name,'delta':delta,'units':args.units,
            'selected_scalars':z[sel].numel(),'changed_scalars':int(torch.count_nonzero(edit)),
            'actual_delta_min':float(edit[sel].min()),'actual_delta_max':float(edit[sel].max()),
            'slice_before':z[args.channel,args.time].tolist(),
            'slice_after':changed[args.channel,args.time].tolist(),
            'decode_seconds':seconds,'rgb_MAE':float(diff.mean()),'rgb_max_abs_diff':float(diff.max()),
            'per_frame_MAE':diff.mean(dim=(0,2,3,4)).tolist()})
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='edits'},indent=2))
    print([(r['delta'],r['rgb_MAE']) for r in report['edits']])

if __name__=='__main__':
    main()
