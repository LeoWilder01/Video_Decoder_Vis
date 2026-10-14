/* Wan 2.1 T2V 1.3B structure only. No numerical tensors are allocated here.
   Shapes include batch; attention matrices denote logical operations, not buffers.
   Sources: Wan-Video/Wan2.1 wan/modules/model.py and official Diffusers config. */
function makeWanArchitecture(latentShape){
 const [C,T,H,W]=latentShape,D=1536,F=8960,L=512,N=T*(H/2)*(W/2),A=12,K=128;
 const nodes=[],edges=[],groups=[],byId=new Map(),span=1020,start=-33000;
 const axis=x=>210+x*(2/9),fmt=s=>'['+s.join(', ')+']';
 function node(id,label,shape,x,offset=0,formula='',group=null,kind='op'){
  const n={id,label,actualShape:shape,shape:[1,1,shape.at(-2)||1,shape.at(-1)||1],type:'architecture',x,y:axis(x)+offset,w:kind==='weight'?28:50,h:kind==='weight'?30:40,formula,group,kind,structural:true};
  nodes.push(n);byId.set(id,n);return n;
 }
 function edge(a,b,label='',kind='flow'){edges.push({from:typeof a==='string'?a:a.id,to:typeof b==='string'?b:b.id,label,kind});}
 function weight(op,suffix,shape,offset=0){const w=node(op.id+'.'+suffix,suffix,shape,op.x+offset,-180+(op.y-axis(op.x)),`Parameter ${fmt(shape)}; not loaded`,op.group,'weight');edge(w,op,'','parameter');return w;}
 function linear(id,label,input,inDim,outDim,x,off,group,seq=N){const n=node(id,label,[1,seq,outDim],x,off,`[1,${seq},${inDim}] × Wᵀ[${inDim},${outDim}] + b[${outDim}]`,group);edge(input,n,'',input.id==='wan.text.2'?'condition':'flow');weight(n,'W',[outDim,inDim]);weight(n,'b',[outDim],35);return n;}
 const noise=node('wan.noise','NOISY LATENT',[1,C,T,H,W],start,0,'z at sampling step s; not the imported clean latent');noise.w=104*.75;noise.h=60*.75;
 const patch=node('wan.patch','PATCH EMBED',[1,D,T,H/2,W/2],start+220,0,'Conv3d kernel/stride [1,2,2]; 16 → 1536');edge(noise,patch);weight(patch,'W',[D,C,1,2,2]);weight(patch,'b',[D],35);
 const tokens=node('wan.tokens','FLATTEN / TRANSPOSE',[1,N,D],start+450,0,`${T} × ${H/2} × ${W/2} = ${N} tokens; no values loaded`);edge(patch,tokens);
 const prompt=node('wan.prompt','PROMPT / TOKENIZER',[1,L],start,-1100,'Text padded/masked to 512 positions');
 const text=node('wan.text_encoder','UMT5-XXL ENCODER',[1,L,4096],start+8300,-2350,'24 encoder blocks; d=4096; 64 heads ×64; gated GELU FFN=10240. Text encoder only.');
 // Text encoder branch: all 24 encoder blocks, with pre-norm attention and gated FFN.
 // They share the scene, above the diffusion backbone; no T5 decoder is involved.
 const lookup=node('wan.text.lookup','TOKEN EMBEDDING',[1,L,4096],start,-2350,'Vocabulary embedding [256384,4096]; padded positions masked');lookup.textDetail=true;edge(prompt,lookup);
 let textLast=lookup;
 for(let i=0;i<24;i++){
  const x=start+180+i*330,base='wan.text.blocks.'+i;
  const tn=(suffix,label,shape,dx,dy,formula)=>{const n=node(base+'.'+suffix,label,shape,x+dx,-2350+dy,formula);n.textDetail=true;return n;};
  const norm=tn('norm1','T5 '+String(i+1).padStart(2,'0')+' / RMS',[1,L,4096],0,0,'T5 RMSNorm, learned gain [4096], no mean subtraction');edge(textLast,norm);
  const qkv=tn('qkv','Q / K / V (each)',[1,64,L,64],45,0,'Three bias-free projections Wq,Wk,Wv [4096,4096]; 64 heads ×64');edge(norm,qkv);
  const scores=tn('attention','QKᵀ + BIAS → SOFTMAX',[1,64,L,L],90,-130,'Relative position bias [64,512,512] + padding mask; softmax over keys');edge(qkv,scores);
  const av=tn('av','A × V → OUT',[1,L,4096],135,0,'Per head [512,512] × [512,64]; merge heads; Wo [4096,4096]');edge(scores,av);edge(qkv,av);
  const r1=tn('residual1','+',[1,L,4096],180,0,'Input + attention output');edge(textLast,r1,'skip','residual');edge(av,r1);
  const n2=tn('norm2','RMS',[1,L,4096],180,180,'T5 RMSNorm with gain [4096]');edge(r1,n2);
  const wi=tn('wi','GATED GELU',[1,L,10240],90,300,'GELU(x Wi_0ᵀ) × (x Wi_1ᵀ); each Wi [10240,4096], no bias');edge(n2,wi);
  const wo=tn('wo','FFN DOWN',[1,L,4096],225,300,'[512,10240] × Woᵀ [10240,4096]; no bias');edge(wi,wo);
  const out=tn('output','T5 '+String(i+1).padStart(2,'0'),[1,L,4096],285,0,'Residual + gated FFN output; inference dropout inactive');edge(wo,out);edge(r1,out,'skip','residual');textLast=out;
 }
 const finalText=node('wan.text.final_norm','FINAL RMS',[1,L,4096],start+180+24*330,-2350,'Final T5 RMSNorm → masked/padded context');finalText.textDetail=true;edge(textLast,finalText);edge(finalText,text);
 const te1=linear('wan.text.0','TEXT LINEAR',text,4096,D,start+8420,-2350,null,L);
 const gelu=node('wan.text.gelu','GELU (tanh)',[1,L,D],start+8530,-2350,'GELU approximation=tanh');edge(te1,gelu);
 const context=linear('wan.text.2','CONTEXT',gelu,D,D,start+8650,-2350,null,L);
 const time=node('wan.time','STEP / SINUSOID',[1,256],start,-700,'Sampling timestep → sinusoidal embedding (256)');
 const e0=linear('wan.time.0','TIME LINEAR',time,256,D,start+170,-700,null,1);e0.actualShape=[1,D];
 const silu=node('wan.time.silu','SiLU',[1,D],start+280,-700,'SiLU');edge(e0,silu);
 const emb=linear('wan.time.2','TIME EMBEDDING',silu,D,D,start+390,-700,null,1);emb.actualShape=[1,D];
 const silu2=node('wan.time.silu2','SiLU',[1,D],start+490,-700,'SiLU');edge(emb,silu2);
 const mod=linear('wan.time.proj','6 × MODULATION',silu2,D,6*D,start+620,-700,null,1);mod.actualShape=[1,6,D];mod.formula='Linear 1536 → 9216; reshape [B,6,1536]';
 let last=tokens;
 for(let i=0;i<30;i++){
  const x=start+1000+i*span,id='wan.blocks.'+i,g={id,index:i,x,y:axis(x),w:980,h:650,entry:last.id};groups.push(g);
  const add=(suffix,label,shape,dx,dy,formula='')=>node(id+'.'+suffix,label,shape,x+dx,dy,formula,id);
  const m=add('modulation','SHIFT / SCALE / GATE',[1,6,D],45,-560,'e = timestep projection + learned [1,6,1536]; six vectors broadcast over tokens');edge(mod,m,'time','condition');weight(m,'learned',[1,6,D]);
  const norm=add('norm1','LN₁',[1,N,D],0,-140,'LayerNorm over 1536 features; no affine parameters');edge(last,norm);
  const ada=add('ada1','SCALE + SHIFT',[1,N,D],60,-140,'LN(x) × (1 + e₁) + e₀');edge(norm,ada);edge(m,ada,'e₀,e₁','condition');
  function attention(prefix,input,base,cross){
   const seq=cross?L:N;
   const q=linear(id+'.'+prefix+'.q','Q',input,D,D,x+base,-280,id);
   const k=linear(id+'.'+prefix+'.k','K',cross?context:input,D,D,x+base,-110,id,seq);
   const v=linear(id+'.'+prefix+'.v','V',cross?context:input,D,D,x+base,60,id,seq);
   const qn=add(prefix+'.norm_q','RMS Q',[1,N,D],base+55,-280,'RMSNorm across all 1536 features, then split 12 heads');edge(q,qn);weight(qn,'gain',[D]);
   const kn=add(prefix+'.norm_k','RMS K',[1,seq,D],base+55,-110,'RMSNorm across all 1536 features, then split 12 heads');edge(k,kn);weight(kn,'gain',[D]);
   const qh=add(prefix+'.qh',cross?'HEADS Q':'3D RoPE Q',[1,A,N,K],base+110,-280,cross?'reshape / transpose → [B,12,N,128]':'reshape / transpose; rotary positions on t,y,x; head dim 128');edge(qn,qh);
   const kh=add(prefix+'.kh',cross?'HEADS K':'3D RoPE K',[1,A,seq,K],base+110,-110,cross?'reshape / transpose → [B,12,512,128]':'reshape / transpose; same 3D RoPE as Q');edge(kn,kh);
   const vh=add(prefix+'.vh','HEADS V',[1,A,seq,K],base+110,60,'reshape / transpose to attention heads');edge(v,vh);
   const score=add(prefix+'.scores','QKᵀ',[1,A,N,seq],base+170,-240,cross?'Q [N,128] × Kᵀ [128,512] / √128':'Q [N,128] × Kᵀ [128,N] / √128; logical matrix, not materialized');edge(qh,score);edge(kh,score);
   const prob=add(prefix+'.softmax','SOFTMAX',[1,A,N,seq],base+220,-240,'softmax over key positions; padding mask; logical attention matrix');edge(score,prob);
   const av=add(prefix+'.av','A × V',[1,A,N,K],base+265,-100,`[N,${seq}] × [${seq},128] → [N,128] per head`);edge(prob,av);edge(vh,av);
   const merge=add(prefix+'.merge','MERGE HEADS',[1,N,D],base+310,-100,'transpose / reshape: 12 ×128 =1536');edge(av,merge);
   return linear(id+'.'+prefix+'.o','OUT PROJ',merge,D,D,x+base+355,-100,id);
  }
  const sa=attention('self_attn',ada,115,false);
  const gate=add('gate1','× GATE',[1,N,D],520,-100,'self attention output × e₂');edge(sa,gate);edge(m,gate,'e₂','condition');
  const r1=add('residual1','+ RESIDUAL',[1,N,D],565,200,'x₁ = x + gate₂ × self_attention');edge(last,r1,'skip','residual');edge(gate,r1);
  // Cross-attention occupies a second lane, retaining the same axis projection.
  const before=nodes.length;
  const norm3=add('norm3','LN₃',[1,N,D],20,620,'Affine LayerNorm (learned gain and bias)');edge(r1,norm3);weight(norm3,'gain',[D]);weight(norm3,'bias',[D],35);
  const ca=attention('cross_attn',norm3,115,true);
  for(const n of nodes.slice(before))if(n.id.includes('.cross_attn.'))n.y+=760;
  const r2=add('residual2','+ RESIDUAL',[1,N,D],565,960,'x₂ = x₁ + cross_attention(LN₃(x₁), context)');edge(r1,r2,'skip','residual');edge(ca,r2);
  const norm2=add('norm2','LN₂',[1,N,D],635,960,'LayerNorm over 1536 features; no affine');edge(r2,norm2);
  const ada2=add('ada2','SCALE + SHIFT',[1,N,D],690,960,'LN₂(x₂) × (1 + e₄) + e₃');edge(norm2,ada2);edge(m,ada2,'e₃,e₄','condition');
  const f1=linear(id+'.ffn.0','FFN UP',ada2,D,F,x+750,760,id);f1.h=95;
  const act=add('ffn.gelu','GELU (tanh)',[1,N,F],810,760,'GELU approximation=tanh');act.h=95;edge(f1,act);
  const f2=linear(id+'.ffn.2','FFN DOWN',act,F,D,x+870,760,id);
  const gate2=add('gate2','× GATE',[1,N,D],920,760,'FFN output × e₅');edge(f2,gate2);edge(m,gate2,'e₅','condition');
  const out=add('output','BLOCK '+String(i+1).padStart(2,'0'),[1,N,D],960,0,'x₃ = x₂ + e₅ × FFN; output to next block');out.w=62;out.h=85;edge(r2,out,'skip','residual');edge(gate2,out);g.exit=out.id;last=out;
 }
 const end=start+1000+30*span;
 const hn=node('wan.head.norm','HEAD LN',[1,N,D],end,0,'LayerNorm; no affine');edge(last,hn);
 const hm=node('wan.head.mod','HEAD MODULATION',[1,2,D],end+110,-340,'time embedding + learned [1,2,1536]; scale and shift');edge(emb,hm,'time','condition');weight(hm,'learned',[1,2,D]);
 const ha=node('wan.head.ada','SCALE + SHIFT',[1,N,D],end+110,0,'LN(x) × (1 + scale) + shift');edge(hn,ha);edge(hm,ha);
 const head=linear('wan.head.proj','OUTPUT LINEAR',ha,D,64,end+230,0,null);
 const pred=node('wan.prediction','UNPATCHIFY',[1,16,T,H,W],end+370,0,'64 =16 ×1 ×2 ×2; predicted flow / velocity, not clean RGB');edge(head,pred);
 const cfg=node('wan.cfg','CFG',[1,16,T,H,W],end+510,0,'v = v_uncond + guidance × (v_cond − v_uncond); same backbone, two text conditions');edge(pred,cfg);
 const scheduler=node('wan.scheduler','SCHEDULER',[1,16,T,H,W],end+670,0,'Solver uses prediction and z_s to produce z_next; repeat until final step');edge(cfg,scheduler);edge(noise,scheduler,'z_s','loop');edge(scheduler,noise,'next sampling step','loop');
 const clean=node('wan.clean','FINAL CLEAN LATENT',[1,16,T,H,W],end+890,0,'Only after the last scheduler step. Imported samples enter here; structure above is not executed.');edge(scheduler,clean,'final step');
 // Assign distinct semantic roles and a declared dimension-to-geometry mapping.
 for(const n of nodes){
  const a=n.actualShape;
  n.role=n.kind==='weight'?'parameter':n.id==='wan.noise'?'video-input':n.id==='wan.prompt'?'text-input':n.id==='wan.time'?'step-input':n.id.startsWith('wan.text.')||n.id==='wan.text_encoder'?'text-feature':n.id.startsWith('wan.time.')?'conditioning':'video-feature';
  n.palette={parameter:'#d9ceba','video-input':'#b9d0d9','text-input':'#d5c7dd','step-input':'#e5d6af','text-feature':'#dfd6e4',conditioning:'#e8ddc3','video-feature':'#cadbd7'}[n.role];
  let channels=1,depth=1,rows=1,cols=a.at(-1),axes='feature';
  if(n.kind==='weight'){
   if(a.length===5){[channels,depth]=a;depth*=a[2];rows=a[3];cols=a[4];axes='kernel y × kernel x; depth = input channel × kernel time';}
   else if(a.length===2){channels=a[0];cols=a[1];axes='one output row × input features';}
   else if(a.length===3){channels=a[1];cols=a[2];axes='one modulation vector × features';}
   else {cols=a[0];axes='parameter vector (not convolution kernels)';}
  }else if(a.length===5){channels=a[1];depth=a[2];rows=a[3];cols=a[4];axes='height × width; depth = video time';}
  else if(a.length===4){channels=a[1];rows=a[2];cols=a[3];axes='query positions × '+(n.id.endsWith('.scores')||n.id.endsWith('.softmax')?'key positions':'head features')+'; copies = heads';}
  else if(a.length===3){rows=a[1];cols=a[2];axes='positions × features';}
  n.shape=[channels,depth,rows,cols];n.viewAxes=axes;
  // Fixed logarithmic ruler preserves ordering while keeping large token matrices in the existing layout.
  const ruler=v=>8+4*Math.log2(Math.max(1,v));
  n.w=ruler(cols);n.h=ruler(rows);
  n.depthSpan=depth>1?4*Math.log2(depth):3;
 }
 // Rigid translation only: every tensor retains its projection and local geometry.
 const left=-12400,pitch=1170,rowPitch=2700,firstY=-8500;
 for(const g of groups){
  const row=Math.floor(g.index/10),col=row%2?9-g.index%10:g.index%10;
  const gx=left+col*pitch,gy=firstY+row*rowPitch+col*pitch*(2/9),dx=gx-g.x,dy=gy-g.y;
  for(const n of nodes)if(n.group===g.id){n.x+=dx;n.y+=dy;}
  Object.assign(g,{x:gx,y:gy,row,col});
 }
 for(const n of nodes){if(n.group)continue;
  const isEnd=n.id.startsWith('wan.head.')||['wan.prediction','wan.cfg','wan.scheduler','wan.clean'].includes(n.id);
  if(isEnd){const last=groups.at(-1);const x0=last.x+1150,y0=last.y+1150*(2/9);n.y+=y0-axis(end);n.x+=x0-end;}
  else {const isText=n.id==='wan.prompt'||n.id==='wan.text_encoder'||n.id.startsWith('wan.text.');
   n.x+=left-900-start;n.y+=(isText?-10200:-9300)-axis(start);
  }
 }
 return {nodes,edges,groups,byId,start:left-900,end:clean,N,D,F,L,heads:A,headDim:K,fmt};
}

function architectureVisible(n){return true;}
function architectureBounds(){const nodes=S.wan.nodes;return {left:Math.min(...nodes.map(n=>n.x))-700,right:Math.max(S.worldWidth,...nodes.map(n=>n.x+n.w))+100,top:Math.min(...nodes.map(n=>n.y))-350,bottom:Math.max(S.worldHeight,...nodes.map(n=>n.y+n.h))+200};}
function architectureLinks(){
 const a=S.wan,selected=S.selected?.id;
 const active=e=>e.from===selected||e.to===selected;

 for(const highlight of [false,true])for(const e of a.edges){
  if(active(e)!==highlight)continue;
  const from=a.byId.get(e.from),to=a.byId.get(e.to);if(!from||!to)continue;
  if(!architectureVisible(from)||!architectureVisible(to))continue;
  if(from.group&&to.group&&from.group===to.group){const gx=a.groups.find(g=>g.id===from.group).x;if(S.ox+(gx+1080)*S.scale<0||S.ox+gx*S.scale>S.width)continue;}
  const color=highlight?'#a86645':e.kind==='residual'?'#71898c':e.kind==='condition'?'#b4b4a0':'#99a79e';
  if(e.kind==='loop'){
   const p=terminal(from),q=terminal(to,false),rail=a.start-420-(e.from==='wan.noise'?70:0);line([p,[p[0]+100,p[1]+180],[rail,p[1]+180],[rail,q[1]-180],[q[0],q[1]-180],q],color,highlight?1.5:.55);
  }else if(from.group&&to.group&&from.group!==to.group){
   const p=terminal(from),q=terminal(to,false),fg=a.groups.find(g=>g.id===from.group),tg=a.groups.find(g=>g.id===to.group);
   if(fg.row!==tg.row){const rail=fg.row%2?a.start-170:Math.max(...a.groups.map(g=>g.x))+1110;line([p,[rail,p[1]],[rail,tg.y-860],[q[0]-30,tg.y-860],[q[0]-30,q[1]],q],color,highlight?1.5:.8);}
   else if(fg.col>tg.col){const yy=tg.y-830;line([p,[p[0]+35,p[1]-820],[q[0]-25,yy],[q[0]-25,q[1]],q],color,highlight?1.5:.7);}
   else flowLink(p,q,highlight);
  }else if(e.kind==='condition'){
   const p=terminal(from),q=terminal(to,false),g=a.groups.find(g=>g.id===to.group);
   if(from.group===to.group){line([p,[q[0],p[1]+(q[0]-p[0])*AXIS_SLOPE],q],color,highlight?1.4:.35);}
   else {const rail=a.start-100-(e.from==='wan.time.proj'?55:0),yy=g?g.y-940:q[1]-300;line([p,[rail,p[1]],[rail,yy],[q[0],yy],q],color,highlight?1.4:.35);}
  }else if(e.kind==='residual'){
   const p=terminal(from),q=terminal(to,false);line([p,[p[0]+12,p[1]+90],[q[0]-12,q[1]+90],q],color,highlight?1.5:.7);
  }else flowLink(terminal(from),terminal(to,false),highlight);
 }
 const clean=a.end,input=S.nodes.find(n=>n.type==='input');flowLink(terminal(clean),[input.x,input.y],S.selected===clean||S.selected===input);
 for(const g of a.groups){const left=S.ox+g.x*S.scale;if(left>S.width||left+1050*S.scale<0)continue;
  const base=g.y;line([[g.x-25,base-820],[g.x+1045,base-820+1070*AXIS_SLOPE],[g.x+1045,base+1160+1070*AXIS_SLOPE],[g.x-25,base+1160],[g.x-25,base-820]],'#bbc5be',.5);
 }
 for(const n of a.nodes){if(!architectureVisible(n))continue;face(n);S.archDrawn.push(n);}
}
function architectureCaptions(){
 const a=S.wan;
 caption('WAN 2.1 / T2V 1.3B',a.start,-10900,{size:15,angle:AXIS_ANGLE});
 caption('30 BLOCKS · 12 HEADS ×128 · STRUCTURE ONLY',a.start,-10900,{size:10,dy:21,angle:AXIS_ANGLE});
 caption('BLUE: NOISE · VIOLET: TEXT · TEAL: FEATURES · OCHRE: FIXED PARAMETERS',a.start,-10900,{size:9,dy:40,angle:AXIS_ANGLE});
 caption('GEOMETRY: LOG₂ DIMENSIONS · PARAMETER COPIES: OUTPUT ROWS / FILTERS',a.start,-10900,{size:9,dy:56,angle:AXIS_ANGLE});
 for(const g of a.groups){
  const out=a.byId.get(g.exit);caption(String(g.index+1).padStart(2,'0'),g.x+490,g.y+490*AXIS_SLOPE+1180,{align:'center',size:10,angle:AXIS_ANGLE});
  if(S.scale>=.10)caption('BLOCK '+String(g.index+1).padStart(2,'0')+' · [1, '+a.N+', 1536]',g.x,g.y-850,{size:10,angle:AXIS_ANGLE});
  if(S.scale>=.13&&S.scale<.65)for(const [dx,dy,label] of [[40,190,'SELF ATTENTION'],[40,890,'CROSS ATTENTION'],[650,1110,'MLP + RESIDUAL']])caption(label,g.x+dx,g.y+dx*AXIS_SLOPE+dy,{size:9,angle:AXIS_ANGLE});
 }
 for(const n of a.nodes){if(!architectureVisible(n)||n.kind==='weight'&&S.scale<1.3)continue;
  if(n.group&&S.scale<.65||n.textDetail&&S.scale<.5&&!n.id.endsWith('.output'))continue;
  if(!n.group&&S.scale<.12)continue;
  caption((n.kind==='weight'?'FIXED / ':n.role.endsWith('input')?'INPUT / ':'')+n.label,n.x,n.y+n.h+12,{size:n.group?9:11,angle:AXIS_ANGLE});
  if(S.scale>=(n.group?.85:.2))caption(a.fmt(n.actualShape),n.x,n.y+n.h+12,{size:9,dy:14,angle:AXIS_ANGLE});
 }
 caption('UMT5-XXL / 24 ENCODER BLOCKS',a.start,-13100,{size:11,angle:AXIS_ANGLE});
 caption('SCHEDULER LOOP · SAME TRANSFORMER AT EACH DENOISING STEP',a.start-420,900,{size:10,angle:AXIS_ANGLE});
 cv.dataset.wanVisibleNodes=S.archDrawn.length;cv.dataset.wanRows=3;cv.dataset.wanBlocks=a.groups.length;cv.dataset.wanTokens=a.N;cv.dataset.wanNodes=a.nodes.length;cv.dataset.wanEdges=a.edges.length;
}
function drawArchitectureLocal(n){
 $('local-view').hidden=false;$('tools').hidden=true;$('local-name').textContent=n.id;
 const canvas=$('local-canvas'),dp=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
 if(canvas.width!==Math.round(w*dp)||canvas.height!==Math.round(h*dp)){canvas.width=Math.round(w*dp);canvas.height=Math.round(h*dp);}
 const q=canvas.getContext('2d');q.setTransform(dp,0,0,dp,0,0);q.clearRect(0,0,w,h);q.fillStyle='#425650';q.font='11px '+MONO;
 const lines=[n.label,n.kind==='weight'?'FIXED PARAMETER · READ ONLY':n.role.toUpperCase(),'shape '+S.wan.fmt(n.actualShape),n.viewAxes,'Representative slice: ['+n.shape.slice(2).join(', ')+']; copies: '+n.shape[0]+'; depth: '+n.shape[1],'',n.formula,'','STRUCTURE ONLY · values not loaded','Geometry uses log₂ dimensions; exact dimensions are shown above.'];
 let y=20;for(const line of lines){let row='';for(const word of line.split(' ')){if(q.measureText(row+word).width>w-30){q.fillText(row,12,y);y+=16;row='';}row+=word+' ';}q.fillText(row,12,y);y+=16;}
 S.localGrid=null;
}
