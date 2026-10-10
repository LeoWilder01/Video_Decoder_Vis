"""Local, decoder-only laboratory. Run: .venv/bin/python server.py"""
import argparse
import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import imageio.v2 as imageio
import numpy as np
import torch
from safetensors import safe_open

from experiment import ROOT, TAEHV, load_decoder
# Use classes from the same imported module as TAEHV (isinstance checks matter).
from taehv import TWorkItem, apply_model_with_memblocks_sequential_single_step

WEB = ROOT / 'web'
RUNS = ROOT / 'experiments' / 'web'
LATENTS = ROOT / 'latents'


def describe(module):
    kind = type(module).__name__
    if kind == 'MemBlock':
        return 'concat(x_t, x_{t-1}) → Conv → ReLU → Conv → ReLU → Conv → +x_t → ReLU'
    if kind == 'Clamp':
        return '3 × tanh(x / 3)'
    if kind == 'TGrow':
        return f'1×1 Conv → reshape；时间 ×{module.stride}'
    if kind == 'Conv2d':
        return f'{module.in_channels}→{module.out_channels}；kernel={list(module.kernel_size)}；stride={list(module.stride)}；padding={list(module.padding)}'
    if kind == 'Upsample':
        return 'nearest；空间 ×2'
    return 'max(0, x)' if kind == 'ReLU' else kind


def topology(shape):
    """Real meta-device execution, including internal MemBlock operations."""
    C,T,H,W = shape
    records = {}
    with torch.device('meta'):
        model = TAEHV(checkpoint_path=None, arch_name='taew2_1')
        def hook(name):
            def capture(module, inputs, output):
                t,c,h,w = output.shape
                records[name] = dict(name=name, kind=type(module).__name__, shape=[c,t,h,w],
                    description=describe(module), parameters=sum(p.numel() for p in module.parameters()),
                    weights=[f'{name}.{n}' for n,_ in module.named_parameters(recurse=False)])
            return capture
        handles=[]
        for name,mod in model.named_modules():
            if name.startswith('decoder.') and not isinstance(mod,(torch.nn.Sequential,torch.nn.Identity)):
                handles.append(mod.register_forward_hook(hook(name)))
        model.decode_video(torch.empty(1,T,C,H,W),parallel=True,show_progress_bar=False)
        for h in handles:
            h.remove()
    # Top-level groups in execution order; internal operations retain actual hook order.
    result=[]
    for i in range(23):
        name=f'decoder.{i}'
        result.append(dict(records[name], children=[v for k,v in records.items() if k.startswith(name+'.')]))
    return result


class Lab:
    def __init__(self, shape=(16,21,60,104), seed=42):
        torch.set_num_threads(4)
        self.model=load_decoder('cpu')
        self.parameters={k:v.detach().numpy() for k,v in self.model.state_dict().items()}
        self.lock=threading.RLock()
        self.worker_lock=threading.Lock()
        self.jobs={}
        self.results=[]
        self.trace=None
        self.history=[]
        self.revision=0
        self.reset(shape,seed)
        RUNS.mkdir(parents=True,exist_ok=True)

    def reset(self,shape,seed):
        if tuple(shape) not in ((16,21,60,104),(16,5,12,20),(16,3,8,8)):
            raise ValueError('Unsupported shape')
        if not 0 <= seed < 2**32:
            raise ValueError('Seed must be in [0, 2^32)')
        with self.lock:
            self.shape=list(shape)
            self.preset='random-42'
            self.seed=seed
            self.z=np.random.default_rng(seed).standard_normal(shape,dtype=np.float32)
            self.baseline=self.z.copy()
            self.revision+=1
            self.history=[]
            self.trace=None
            self.graph=topology(shape)

    def presets(self):
        items=[dict(id='random-42',label='Random · seed 42')]
        for path in sorted(LATENTS.glob('*.safetensors')):
            items.append(dict(id=path.name,label=path.stem))
        return items

    def select_preset(self, name):
        if not self.worker_lock.acquire(blocking=False):
            raise ValueError('Wait for the current decode before switching samples')
        try:
            if name not in [p['id'] for p in self.presets()]:raise ValueError('Unknown sample')
            shape=(16,21,60,104)
            if name=='random-42':
                z=np.random.default_rng(42).standard_normal(shape,dtype=np.float32)
            else:
                with safe_open(str(LATENTS/name),framework='np') as f:
                    meta=f.metadata() or {}
                    if any(meta.get(k)!=v for k,v in dict(layout='CTHW',model_family='wan2.1',stage='final_clean',latent_space='diffusion').items()):
                        raise ValueError('Expected Wan 2.1 final_clean diffusion-space CTHW metadata')
                    z=f.get_tensor('latent').astype(np.float32)
                if z.shape!=shape or not np.isfinite(z).all():raise ValueError('Expected finite [16,21,60,104] latent')
            with self.lock:
                if tuple(self.shape)!=shape:self.graph=topology(shape)
                self.shape=list(shape);self.z=z.copy();self.baseline=z.copy()
                self.seed=42 if name=='random-42' else None
                self.preset=name;self.revision+=1;self.history=[];self.trace=None
                return self.status()
        finally:
            self.worker_lock.release()

    def lens_patch(self, name, c, t, y, x, revision):
        """Return only a 3x3 neighborhood in adjacent slices; never compute a trace."""
        with self.lock:
            if revision != self.revision:
                raise ValueError('Stale revision')
            result=next((r for r in reversed(self.results) if r['revision']==revision),None)
            layers=[]
            for depth in range(max(0,t-1),t+2):
                plane=None
                if name.startswith('z.'):
                    if 0<=c<self.z.shape[0] and depth<self.z.shape[1]:plane=self.z[c,depth]
                elif name in self.parameters:
                    a=self.parameters[name]
                    a=a.reshape((*a.shape,)+(1,)*(4-a.ndim))
                    if 0<=c<a.shape[0] and depth<a.shape[1]:plane=a[c,depth]
                elif result and name=='rgb':
                    if depth<result['shape'][1] and 0<=c<3:plane=np.load(RUNS/(result['id']+'.npy'),mmap_mode='r')[depth,c]
                elif result and c==0 and name in result.get('activations',{}):
                    if depth<result['activations'][name]['shape'][0]:plane=np.load(RUNS/(result['id']+'.'+name+'.npy'),mmap_mode='r')[depth]
                if plane is None:
                    trace=self.trace
                    if trace and trace['meta']['revision']==revision and trace['meta']['layer']==name and trace['meta']['time']==depth and 0<=c<trace['array'].shape[0]:plane=trace['array'][c]
                if plane is not None and 0<=y<plane.shape[0] and 0<=x<plane.shape[1]:
                    layers.append(dict(t=depth,values=plane[y:y+3,x:x+3].tolist()))
            return dict(revision=revision,layers=layers)

    def status(self):
        with self.lock:
            return dict(shape=self.shape,seed=self.seed,preset=self.preset,revision=self.revision,
                changed_scalars=int(np.count_nonzero(self.z!=self.baseline)),
                history=self.history[-16:],jobs=list(self.jobs.values())[-6:],results=self.results,
                trace=None if self.trace is None else self.trace['meta'])

    def edit(self,body):
        with self.lock:
            if int(body['revision'])!=self.revision:
                raise ValueError('Latent revision changed. Reload before editing.')
            c,t,y,x=[int(body[k]) for k in ('c','t','y','x')]
            if not all(0<=i<n for i,n in zip((c,t,y,x),self.shape)):
                raise ValueError('Index out of bounds')
            value=float(body['value'])
            if not math.isfinite(value) or abs(value)>1e6:
                raise ValueError('Value must be finite and within ±1e6')
            mode=body.get('mode','scalar')
            if mode=='scalar':
                sel=(slice(c,c+1),slice(t,t+1),slice(y,y+1),slice(x,x+1))
            elif mode=='channel':
                sel=(slice(c,c+1),slice(None),slice(None),slice(None))
            elif mode in ('region','time'):
                dt,dy,dx=[int(v) for v in body.get('extent',[1,1,1])]
                if min(dt,dy,dx)<1 or t+dt>self.shape[1]:
                    raise ValueError('Invalid time interval')
                if mode=='time':
                    sel=(slice(None),slice(t,t+dt),slice(None),slice(None))
                else:
                    if y+dy>self.shape[2] or x+dx>self.shape[3]:
                        raise ValueError('Region out of bounds')
                    sel=(slice(c,c+1),slice(t,t+dt),slice(y,y+dy),slice(x,x+dx))
            else:
                raise ValueError('Unknown mode')
            before=float(self.z[c,t,y,x])
            if mode=='scalar':
                self.z[sel]=value
            else:
                if np.max(np.abs(self.z[sel]))+abs(value)>1e6:
                    raise ValueError('Result exceeds ±1e6')
                self.z[sel]+=value
            self.revision+=1
            self.history.append(dict(revision=self.revision,mode=mode,index=[c,t,y,x],
                value=value,before=before,after=float(self.z[c,t,y,x]),count=self.z[sel].size))
            return self.status()

    def start(self,kind,body):
        if not self.worker_lock.acquire(blocking=False):
            raise ValueError('Decoder is busy; wait for the current computation.')
        try:
            with self.lock:
                z=self.z.copy();revision=self.revision
                job_id=uuid.uuid4().hex[:12]
                job=dict(id=job_id,kind=kind,state='running',progress=0,revision=revision,started=time.time())
                if kind=='trace':
                    records=[r for group in self.graph for r in [group,*group['children']]]
                    target=next((r for r in records if r['name']==body.get('layer')),None)
                    if target is None or not 0<=int(body.get('time',0))<target['shape'][1]:
                        raise ValueError('Invalid trace layer/time')
                    body=dict(body,target=target)
                self.jobs[job_id]=job
            thread=threading.Thread(target=self.work,args=(job,z,body),daemon=True)
            thread.start()
            return dict(job=job)
        except Exception:
            self.worker_lock.release()
            raise

    def work(self,job,z,body):
        start=time.perf_counter()
        handles=[]
        last_update=[0.]
        def location(index):
            def record(module, inputs):
                now=time.monotonic()
                if now-last_update[0]>.08:
                    job['current_layer']=f'decoder.{index}'
                    last_update[0]=now
            return record
        for index,module in enumerate(self.model.decoder):
            handles.append(module.register_forward_pre_hook(location(index)))
        try:
            with torch.inference_mode():
                if job['kind']=='decode':
                    self.decode(job,z)
                else:
                    self.capture(job,z,body)
            job.update(state='done',progress=1,seconds=time.perf_counter()-start)
        except Exception as error:
            job.update(state='error',error=str(error))
        finally:
            for handle in handles:handle.remove()
            self.worker_lock.release()

    def queue(self,z):
        x=torch.from_numpy(z).permute(1,0,2,3).unsqueeze(0)
        return [TWorkItem(xt,0) for xt in x.unbind(1)]

    def decode(self,job,z):
        C,T,H,W=z.shape
        frames=4*T-3
        stem=RUNS/job['id']
        storage=np.lib.format.open_memmap(str(stem)+'.npy',mode='w+',dtype=np.float32,shape=(frames,3,H*8,W*8))
        queue=self.queue(z);memory=[None]*len(self.model.decoder)
        writer=imageio.get_writer(str(stem)+'.mp4',fps=16,macro_block_size=1,codec='libx264',quality=7)
        n=0
        captures={};counts={};handles=[]
        records=[r for group in self.graph for r in [group,*group['children']]]
        for record in records:
            name=record['name'];_,times,h,w=record['shape']
            # Main slabs retain channel 0 across time; small internal slabs retain t=0.
            saved_times=times if name.count('.')==1 else 1
            captures[name]=np.lib.format.open_memmap(str(stem)+'.'+name+'.npy',mode='w+',dtype=np.float32,shape=(saved_times,h,w))
            counts[name]=0
            def capture(module,inputs,output,name=name):
                start=counts[name];batch=output.shape[0];target=captures[name]
                take=min(batch,len(target)-start)
                if take>0:target[start:start+take]=output[:take,0].detach().cpu().numpy()
                counts[name]+=batch
            handles.append(self.model.get_submodule(name).register_forward_hook(capture))
        try:
            while queue:
                rgb=apply_model_with_memblocks_sequential_single_step(self.model.decoder,memory,queue)
                if rgb is None:
                    break
                if n>=3:
                    frame=rgb[0,0].clamp(0,1).numpy()
                    storage[n-3]=frame
                    writer.append_data((frame.transpose(1,2,0)*255).round().astype(np.uint8))
                n+=1
                job['progress']=n/(frames+3)
        finally:
            for handle in handles:handle.remove()
            for arr in captures.values():arr.flush()
            writer.close();storage.flush()
        if n!=frames+3:
            raise RuntimeError(f'Wrong frame count {n}')
        meta=dict(id=job['id'],revision=job['revision'],shape=[3,frames,H*8,W*8],fps=16,
            activations={name:dict(shape=list(arr.shape),channel=0) for name,arr in captures.items()},
            seconds=time.time()-job['started'],video=f'/runs/{job["id"]}.mp4',synthetic=True)
        (Path(str(stem)+'.json')).write_text(json.dumps(meta))
        with self.lock:
            self.results.append(meta)
            # Keep only two generated runs and their exact RGB arrays on disk.
            while len(self.results)>2:
                old=self.results.pop(0)
                for name in old.get('activations',{}):
                    (RUNS/(old['id']+'.'+name+'.npy')).unlink(missing_ok=True)
                for suffix in ('.mp4','.npy','.json'):
                    (RUNS/(old['id']+suffix)).unlink(missing_ok=True)

    def capture(self,job,z,body):
        name=body['target']['name'];target_time=int(body.get('time',0))
        module=self.model.get_submodule(name)
        count=0;captured=None;arithmetic=None
        class Finished(Exception):
            pass
        def hook(mod,inputs,output):
            nonlocal count,captured,arithmetic
            batch=output.shape[0]
            if count<=target_time<count+batch:
                # Clone before any subsequent inplace ReLU can mutate the output.
                captured=output[target_time-count].detach().cpu().numpy().copy()
                if isinstance(mod,torch.nn.Conv2d):
                    # A real dot product at output channel 0, y=x=0, including padding.
                    frame=inputs[0][target_time-count]
                    py,px=mod.padding;kh,kw=mod.kernel_size
                    padded=torch.nn.functional.pad(frame,(px,px,py,py))
                    patch=padded[:,:kh,:kw]
                    kernel=mod.weight[0]
                    products=patch*kernel
                    bias=float(mod.bias[0]) if mod.bias is not None else 0.
                    arithmetic=dict(input_patch=patch[0].tolist(),kernel=kernel[0].tolist(),
                        products=products[0].tolist(),channel_sum=float(products[0].sum()),
                        all_channel_sum=float(products.sum()),bias=bias,
                        output=float(captured[0,0,0]),input_channels=int(frame.shape[0]),
                        location=[0,target_time,0,0])
                raise Finished()
            count+=batch
            job['progress']=min(0.99,count/max(1,target_time+1))
        handle=module.register_forward_hook(hook)
        try:
            # Later blocks cannot affect this activation; avoid decoding full RGB
            # for preceding timesteps just to inspect an early layer.
            prefix=self.model.decoder[:int(name.split('.')[1])+1]
            queue=self.queue(z);memory=[None]*len(prefix)
            while queue:
                apply_model_with_memblocks_sequential_single_step(prefix,memory,queue)
        except Finished:
            pass
        finally:
            handle.remove()
        if captured is None:
            raise RuntimeError('Activation was not captured')
        meta=dict(id=job['id'],layer=name,time=target_time,shape=list(captured.shape),revision=job['revision'],
            minimum=float(captured.min()),maximum=float(captured.max()),mean=float(captured.mean()),
            std=float(captured.std()),seconds=time.time()-job['started'],arithmetic=arithmetic)
        with self.lock:
            self.trace=dict(meta=meta,array=captured)


class Handler(BaseHTTPRequestHandler):
    lab=None

    def log_message(self,format,*args):
        if ' /api/status ' not in str(args):
            super().log_message(format,*args)

    def send(self,body,status=200,kind='application/json',extra=None):
        if kind=='application/json':
            body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if extra:
            for k,v in extra.items():self.send_header(k,v)
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

    def array(self,array,extra=None):
        self.send(np.asarray(array,dtype='<f4').tobytes(),kind='application/octet-stream',
            extra={'X-Shape':json.dumps(list(array.shape)),**(extra or {})})

    def trusted(self):
        hosts=(f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        if self.headers.get('Host') not in hosts:
            return False
        origin=self.headers.get('Origin')
        return not origin or origin in tuple('http://'+h for h in hosts)

    def do_GET(self):
        if not self.trusted():return self.send({'error':'Local access only'},403)
        url=urlparse(self.path);q={k:v[0] for k,v in parse_qs(url.query).items()};p=url.path
        try:
            if p=='/api/lens':return self.send(self.lab.lens_patch(q['name'],int(q['c']),int(q['t']),int(q['y']),int(q['x']),int(q['revision'])))
            if p=='/api/presets':return self.send(self.lab.presets())
            if p=='/api/status':return self.send(self.lab.status())
            if p=='/api/model':
                inventory=json.loads((ROOT/'reports/weights-inventory.json').read_text())
                return self.send(dict(stages=self.lab.graph,weights=[x for x in inventory if x['name'].startswith('decoder.')],
                    parameter_count=9844611,device='CPU / float32',weight_precision='FP16 source',top_level_operations=23))
            if p=='/api/latent':
                with self.lab.lock:return self.array(self.lab.z,{'X-Revision':str(self.lab.revision)})
            if p=='/api/baseline':
                with self.lab.lock:return self.array(self.lab.baseline)
            if p=='/api/latent.npy':
                import io
                buffer=io.BytesIO()
                with self.lab.lock:np.save(buffer,self.lab.z,allow_pickle=False)
                return self.send(buffer.getvalue(),kind='application/octet-stream',extra={'Content-Disposition':'attachment; filename="latent-CTHW.npy"'})
            if p=='/api/weight-previews':
                return self.send({name:dict(shape=list(a.shape), values=a.reshape(-1)[:48].tolist()) for name,a in self.lab.parameters.items()})
            if p=='/api/weight':return self.array(self.lab.parameters[q['name']])
            if p=='/api/decode-activation':
                with self.lab.lock:
                    result=next((r for r in self.lab.results if r['id']==q.get('id')),None)
                    name=q.get('layer');t=int(q.get('time',0))
                    if result is None or name not in result.get('activations',{}):raise ValueError('Activation not recorded')
                    if not 0<=t<result['activations'][name]['shape'][0]:raise ValueError('Time not recorded')
                    return self.array(np.load(RUNS/(result['id']+'.'+name+'.npy'),mmap_mode='r')[t])
            if p=='/api/activation':
                with self.lab.lock:
                    trace=self.lab.trace
                    if trace is None or q.get('id')!=trace['meta']['id']:raise ValueError('Trace expired; compute again')
                    c=int(q.get('c',0))
                    if not 0<=c<trace['array'].shape[0]:raise ValueError('Channel out of bounds')
                    return self.array(trace['array'][c])
            if p=='/api/rgb':
                with self.lab.lock:
                    result=next((r for r in self.lab.results if r['id']==q.get('id')),None)
                    if result is None:raise ValueError('Result not found')
                    frame=int(q.get('frame',0))
                    if not 0<=frame<result['shape'][1]:raise ValueError('Frame out of bounds')
                    arr=np.load(RUNS/(result['id']+'.npy'),mmap_mode='r')[frame]
                    return self.array(arr)
            if p.startswith('/runs/'):
                filename=p.removeprefix('/runs/')
                if not any(filename==r['id']+'.mp4' for r in self.lab.results):raise ValueError('Video not found')
                return self.video(RUNS/filename)
            files={'/':'index.html','/app.js':'app.js','/full-model/':'full-model/index.html','/full-model':'full-model/index.html','/full-model/app.js':'full-model/app.js','/full-model/wan-architecture.js':'full-model/wan-architecture.js','/style.css':'style.css'}
            if p not in files:return self.send({'error':'Not found'},404)
            kinds={'.html':'text/html; charset=utf-8','.js':'application/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}
            file=WEB/files[p]
            return self.send(file.read_bytes(),kind=kinds[file.suffix])
        except (ValueError,KeyError,IndexError) as error:
            return self.send({'error':str(error)},400)
        except Exception as error:
            return self.send({'error':str(error)},500)

    def video(self,path):
        size=path.stat().st_size;start=0;end=size-1;code=200
        request=self.headers.get('Range')
        if request:
            import re
            match=re.fullmatch(r'bytes=(\d+)-(\d*)',request)
            if not match:return self.send({'error':'Unsupported range'},416)
            start=int(match[1]);end=min(size-1,int(match[2])) if match[2] else size-1
            if start>end:return self.send({'error':'Invalid range'},416)
            code=206
        with path.open('rb') as f:
            f.seek(start);data=f.read(end-start+1)
        extra={'Accept-Ranges':'bytes'}
        if code==206:extra['Content-Range']=f'bytes {start}-{end}/{size}'
        self.send(data,status=code,kind='video/mp4',extra=extra)

    def do_POST(self):
        if not self.trusted():return self.send({'error':'Local access only'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>16384 or self.headers.get('Content-Type')!='application/json':
                raise ValueError('Expected a small JSON request')
            body=json.loads(self.rfile.read(size));p=urlparse(self.path).path
            if p=='/api/preset':return self.send(self.lab.select_preset(body.get('id')))
            if p=='/api/edit':return self.send(self.lab.edit(body))
            if p=='/api/reset':
                if self.lab.worker_lock.locked():raise ValueError('Wait for active computation before resetting')
                self.lab.reset(body.get('shape',self.lab.shape),int(body.get('seed',42)))
                return self.send(self.lab.status())
            if p=='/api/restore':
                with self.lab.lock:
                    self.lab.z=self.lab.baseline.copy();self.lab.revision+=1;self.lab.history=[]
                return self.send(self.lab.status())
            if p in ('/api/decode','/api/trace'):
                return self.send(self.lab.start(p.rsplit('/',1)[-1],body))
            return self.send({'error':'Not found'},404)
        except (ValueError,KeyError,TypeError) as error:
            self.send({'error':str(error)},400)
        except Exception as error:
            self.send({'error':str(error)},500)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--small',action='store_true')
    args=parser.parse_args()
    Handler.lab=Lab(shape=(16,5,12,20) if args.small else (16,21,60,104))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Wan latent laboratory: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
