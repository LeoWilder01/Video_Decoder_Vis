'use strict';
const $=id=>document.getElementById(id), cv=$('scene'),ctx=cv.getContext('2d');
const S={model:null,z:null,shape:null,revision:0,nodes:[],selected:null,scale:1,ox:0,oy:0,width:0,height:0,t:0,c:0,y:0,x:0,weights:new Map(),samples:{},activations:new Map(),rgb:null,result:null,job:null,request:0,dirty:true,zoomLevel:0,focusNode:null,drag:null,edit:null,done:new Set()};
const MONO='"SFMono-Regular",Consolas,"Liberation Mono",monospace';
const AXIS_SLOPE=2/9,AXIS_ANGLE=Math.atan(AXIS_SLOPE),START_X=820,START_Y=555,STEP=150;
const axisY=(x,base)=>base+(x-START_X)*AXIS_SLOPE;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,Number(v)||0));
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d;}
async function binary(path,signal){const r=await fetch(path,{signal});if(!r.ok)throw Error((await r.json()).error);return {data:new Float32Array(await r.arrayBuffer()),shape:JSON.parse(r.headers.get('X-Shape'))};}
function fail(e){$('error').textContent=e.message||String(e);}
function action(id,fn){$(id).addEventListener('click',()=>Promise.resolve().then(fn).catch(fail));}
function index(c,t,y,x){const [,T,H,W]=S.shape;return ((c*T+t)*H+y)*W+x;}
function build(){S.nodes=[];for(let c=0;c<16;c++){const col=c%4,row=Math.floor(c/4);S.nodes.push({id:'z.'+c,type:'input',c,x:85+col*110,y:80+col*110*AXIS_SLOPE+row*175,w:90,h:90*S.shape[3]/S.shape[2],shape:S.shape});}
 let stageX=START_X;for(const [i,g] of S.model.stages.entries()){const x=stageX,y=axisY(x,START_Y),w=g.shape[2]*.75,h=g.shape[3]*.75;stageX+=Math.max(STEP,w*.72+50);S.nodes.push({id:g.name,type:'activation',info:g,x,y,w,h,shape:g.shape});const ws=S.model.weights.filter(w=>w.name.startsWith(g.name+'.')).sort((a,b)=>a.name.replace(/\.(weight|bias)$/,'').localeCompare(b.name.replace(/\.(weight|bias)$/,''),undefined,{numeric:true})||(a.name.endsWith('.weight')?-1:1));ws.forEach((w,k)=>{const dx=(k%2)*76;S.nodes.push({id:w.name,type:'weight',info:w,x:x+dx,y:y-350+Math.floor(k/2)*82,w:66,h:47,shape:w.shape});});g.children.forEach((ch,k)=>S.nodes.push({id:ch.name,type:'activation',info:ch,x:x+k*20,y:y-77+k*20*AXIS_SLOPE,w:16,h:26,shape:ch.shape,child:true}));}
 const x=stageX+100;S.nodes.push({id:'RGB',type:'output',x,y:axisY(x,START_Y),w:S.shape[2]*8*.75,h:S.shape[3]*8*.75,shape:[3,S.shape[1]*4-3,S.shape[2]*8,S.shape[3]*8]});S.worldWidth=x+S.shape[2]*8*.75+250;S.worldHeight=axisY(x,START_Y)+S.shape[3]*8*.75+520;S.railBase=START_Y+S.shape[3]*8*.75+390;for(const n of S.nodes){if((n.type==='activation'&&!n.child)||n.type==='output')n.y=axisY(n.x+.72*n.w,START_Y-100)+.38*n.w;}for(const n of S.nodes){if(n.type==='weight'&&n.id.endsWith('.bias')){const w=S.nodes.find(v=>v.id===n.id.replace(/\.bias$/,'.weight'));if(w)n.y=w.y+(n.x-w.x)*AXIS_SLOPE;}}// Turn data faces to landscape while preserving each front top-right anchor.
 for(const n of S.nodes){if(n.type==='weight')continue;const ax=n.x+.72*n.w,ay=n.y-.38*n.w;[n.w,n.h]=[n.h,n.w];n.x=ax-.72*n.w;n.y=ay+.38*n.w;
  if(n.type==='input'){const col=n.c%4,row=Math.floor(n.c/4);n.x=col*160;n.y=210+col*160*AXIS_SLOPE+row*125;}
 }
 const output=S.nodes.at(-1);output.channel=0;
 for(let c=1;c<3;c++)S.nodes.push({...output,id:'RGB.'+c,channel:c,x:output.x-c*54.72,y:output.y-c*54.72*AXIS_SLOPE});
 // Translate each internal group as a rigid unit; keep the main stages fixed.
 const stages=S.nodes.filter(n=>n.type==='activation'&&!n.child);
 for(let i=1;i<stages.length;i++){
  const stage=stages[i],previous=stages[i-1],children=stage.info.children.map(ch=>S.nodes.find(n=>n.id===ch.name));if(!children.length)continue;
  // Measure front/back face positions along the depth axis, not screen-x bounds.
  const bounds=children.map(faceDepthBounds),groupCenter=(Math.min(...bounds.map(b=>b.back))+Math.max(...bounds.map(b=>b.front)))/2;
  const gapCenter=(faceDepthBounds(previous).front+faceDepthBounds(stage).back)/2,dx=gapCenter-groupCenter,dy=dx*AXIS_SLOPE;
  const weights=S.nodes.filter(n=>n.type==='weight'&&n.id.startsWith(stage.id+'.'));
  for(const n of [...children,...weights]){n.x+=dx;n.y+=dy;}
 }
 const tip=output.x+.72*output.w;S.videoPanel={x:tip-1990,y:-180-290*AXIS_SLOPE,w:1450,h:1450*S.shape[2]/S.shape[3]};
 S.worldWidth=Math.max(S.worldWidth,tip+450);S.copies=new Map();S.dirty=true;}
const rootOf=n=>n?.root||n;
const sameGroup=(a,b)=>!!a&&!!b&&rootOf(a)===rootOf(b);
function channelNode(n,c){n=rootOf(n);if(n.type==='input')return n;const key=n.id+':'+c;let q=S.copies.get(key);if(!q){const C=dimensions(n)[0],rows=Math.min(8,C),cols=Math.ceil(C/rows),column=c%cols,row=Math.floor(c/cols),direction=n.type==='weight'?1:-1,g=depthGeometry(n),du=direction*column*(n.w+Math.abs(g.du)+14),dv=-direction*row*(n.h+Math.abs(g.dv)+14);q={...n,root:n,channel:c,x:n.x+.72*du,y:n.y-.38*du+dv};S.copies.set(key,q);}return q;}
function dimensions(n){if(n.type==='weight'){const [O,I=1,H=1,W=1]=n.shape;return [O,I,H,W];}return n.shape;}
function sliceIndex(n){if(n.type==='output'&&n!==S.selected)return S.rgbFrame||0;return sameGroup(n,S.selected)||(n.type==='input'&&S.selected?.type==='input')?clamp(S.t,0,dimensions(n)[1]-1):0;}
function faceDepthBounds(n){const front=(n.y+(.38/.72)*n.x)/(AXIS_SLOPE+.38/.72);return {front,back:front+.72*depthGeometry(n).du};}
function depthGeometry(n){if(n.structural)return {count:2,du:-12,dv:-12*(.38+.72*AXIS_SLOPE)};const count=dimensions(n)[1],span=count>1?(n.type==='activation'?count*(38/42):Math.min(n.w*.65,n.type==='weight'?24:n.type==='input'?60:38,Math.max(24,count*2))):0;return {count,du:-span,dv:-span*(.38+.72*AXIS_SLOPE)};}
function sectionOrigin(n,t=sliceIndex(n)){const g=depthGeometry(n),f=g.count>1?t/(g.count-1):0;return {u:g.du*f,v:g.dv*f};}
function localPoint(x,y,n,t=sliceIndex(n)){const o=sectionOrigin(n,t),u=(x-n.x)/.72-o.u,v=y-n.y+.38*((x-n.x)/.72)-o.v;return {u,v};}
function dataFor(n,requestedTime){if(n.structural)return null;if(S.slicePending&&(sameGroup(n,S.selected)||(n.type==='input'&&S.selected?.type==='input')))return null;const [C,T,H,W]=dimensions(n);let c=n.type==='input'?n.c:(n.channel??(sameGroup(n,S.selected)?clamp(S.c,0,C-1):0)),t=requestedTime??sliceIndex(n);if(n.type==='input')return {data:S.z.subarray(index(c,t,0,0),index(c,t,0,0)+H*W),h:H,w:W,rev:S.revision};if(n.type==='weight'){const full=S.weights.get(n.id);if(full)return {data:full.subarray((c*T+t)*H*W,(c*T+t+1)*H*W),h:H,w:W};const a=c===0&&t===0?S.samples[n.id]:null;return a?{data:a.values.slice(0,H*W),h:H,w:W,sample:true}:null;}if(n.type==='output')return S.rgb&&t===S.rgbFrame?{data:S.rgb.data.subarray(c*H*W,(c+1)*H*W),h:H,w:W,rev:S.result.revision}:null;const saved=S.decodePlanes?.get(n.id+':'+t);if(c===0&&saved&&saved.revision===S.revision)return {data:saved.data,h:H,w:W,rev:saved.revision};const base=S.activations.get(n.id),cached=S.activationPlanes?.get(n.id+':'+c+':'+t),a=cached?.id===base?.id?cached:base;return a&&a.c===c&&a.t===t?{data:a.data,h:a.shape[0],w:a.shape[1],rev:a.revision}:null;}
// Expand scientific notation without changing the underlying value.
function decimalValue(v){const [mantissa,exponent]=String(v).toLowerCase().split('e');if(exponent===undefined)return mantissa;const sign=mantissa.startsWith('-')?'-':'',parts=mantissa.replace('-','').split('.'),digits=parts.join(''),point=parts[0].length+Number(exponent);return sign+(point<=0?'0.'+'0'.repeat(-point)+digits:point>=digits.length?digits+'0'.repeat(point-digits.length):digits.slice(0,point)+'.'+digits.slice(point));}
function displayValue(v){return Math.abs(v)>0&&Math.abs(v)<.01?decimalValue(Number(v.toPrecision(2))):v.toFixed(2);}
function rgb(v,limit=3){const f=Math.min(1,Math.abs(v)/limit),t=v<0?[102,153,158]:[206,143,118];return t.map(n=>Math.round(248+(n-248)*f));}
function layout(n){const [, ,H,W]=dimensions(n);return {rows:H,cols:W,transpose:false};}
function address(n,row,col){return [row,col];}
function texture(data,transpose=false){const out=document.createElement('canvas');out.width=transpose?data.h:data.w;out.height=transpose?data.w:data.h;const x=out.getContext('2d'),im=x.createImageData(out.width,out.height);for(let r=0;r<out.height;r++)for(let c=0;c<out.width;c++){const i=transpose?c*data.w+r:r*data.w+c;const col=data.rgb?[0,1,2].map(ch=>Math.round(data.data[ch*data.w*data.h+i]*255)):rgb(data.data[i]);im.data.set([...col,255],(r*out.width+c)*4);}x.putImageData(im,0,0);return out;}
const texCache=new Map(),pendingWeights=new Set();
const textureBuffers=new WeakMap();let textureBufferId=0,textureBytes=0;
function clearTextures(){texCache.clear();textureBytes=0;}
function cachedTexture(n,d,transpose){
 const source=d.data.buffer||S.samples[n.id];if(!textureBuffers.has(source))textureBuffers.set(source,++textureBufferId);
 const key=[textureBuffers.get(source),d.data.byteOffset||0,d.h,d.w,transpose,d.rev??0,d.sample?n.id:''].join(':');
 let tex=texCache.get(key);if(tex){texCache.delete(key);texCache.set(key,tex);return tex;}
 tex=texture(d,transpose);texCache.set(key,tex);textureBytes+=tex.width*tex.height*4;
 while(texCache.size>512||textureBytes>128*1024*1024){const oldest=texCache.keys().next().value,old=texCache.get(oldest);textureBytes-=old.width*old.height*4;texCache.delete(oldest);}
 cv.dataset.textureBuilds=String(Number(cv.dataset.textureBuilds||0)+1);return tex;
}
let sliceTimer=null,sliceController=null;
function cancelSliceLoad(){clearTimeout(sliceTimer);sliceController?.abort();sliceController=null;S.slicePending=false;}
function scheduleSliceLoad(){
 cancelSliceLoad();const n=S.selected;if(!n)return;S.slicePending=true;S.rgbRequest=(S.rgbRequest||0)+1;controls();
 sliceTimer=setTimeout(async()=>{S.slicePending=false;if(S.selected!==n)return;const controller=new AbortController();sliceController=controller;
  try{if(n.type==='activation')await loadActivation(n,controller.signal);else if(n.type==='output')await loadRGB(controller.signal);}
  catch(e){if(e.name!=='AbortError')fail(e);}
  finally{if(sliceController===controller){sliceController=null;controls();}}
 },160);
}
function face(n){
 const bounds=depthGeometry(n);if(S.ox+(n.x+.72*n.w)*S.scale<0||S.ox+(n.x+.72*bounds.du)*S.scale>S.width||S.oy+(n.y+n.h)*S.scale<0||S.oy+(n.y-.38*(n.w+bounds.du)+bounds.dv)*S.scale>S.height){n.grid=null;return;}
 const numberLabels=[];const selected=S.selected===n,d=dataFor(n),g=layout(n),active=sectionOrigin(n),geometry=depthGeometry(n);
 const layers=S.scale>1.4?geometry.count:Math.min(4,geometry.count);
 ctx.save();ctx.translate(n.x,n.y);ctx.transform(.72,-.38,0,1,0,0);
 // Opaque cut volume: the back and all four thickness faces occlude geometry behind it.
 const du=geometry.du,dv=geometry.dv,au=active.u,av=active.v;
 ctx.fillStyle='#eff3ed';ctx.strokeStyle='#a5b3a8';ctx.lineWidth=.5/S.scale;
 ctx.fillRect(du,dv,n.w,n.h);ctx.strokeRect(du,dv,n.w,n.h);
 const cap=(points,fill)=>{ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.stroke();};
 cap([[du+n.w,dv],[du+n.w,dv+n.h],[au+n.w,av+n.h],[au+n.w,av]],'#dce5db');
 cap([[du,dv+n.h],[du+n.w,dv+n.h],[au+n.w,av+n.h],[au,av+n.h]],'#e3ebe2');
 cap([[du,dv],[du+n.w,dv],[au+n.w,av],[au,av]],'#edf2eb');
 cap([[du,dv],[au,av],[au,av+n.h],[du,dv+n.h]],'#e3ebe2');
 if(S.scale>=1.4){ctx.strokeStyle='#98aaa480';for(let j=layers-1;j>=0;j--){const t=layers>1?j*(geometry.count-1)/(layers-1):0;if(t<sliceIndex(n))continue;const o=sectionOrigin(n,t);ctx.beginPath();ctx.moveTo(o.u,o.v+n.h);ctx.lineTo(o.u,o.v);ctx.lineTo(o.u+n.w,o.v);ctx.stroke();}}
 ctx.translate(active.u,active.v);ctx.fillStyle=n.info?.kind==='TGrow'||n.info?.kind==='Upsample'?'#e6d6cc':'#dde8e2';ctx.fillRect(0,0,n.w,n.h);
 if(d){if(n.child){ctx.fillStyle='rgb('+rgb(d.data[0]).join(',')+')';ctx.fillRect(0,0,n.w,n.h);}else{const tex=cachedTexture(n,d,g.transpose);ctx.imageSmoothingEnabled=false;ctx.drawImage(tex,0,0,n.w,n.h);}}
 const cw=n.w/g.cols,ch=n.h/g.rows,cellWidth=cw*.72*S.scale,cellHeight=ch*S.scale;
 const corners=[[0,0],[S.width,0],[0,S.height],[S.width,S.height]].map(([sx,sy])=>localPoint((sx-S.ox)/S.scale,(sy-S.oy)/S.scale,n));
 const c0=clamp(Math.floor(Math.min(...corners.map(p=>p.u))/cw),0,g.cols),c1=clamp(Math.ceil(Math.max(...corners.map(p=>p.u))/cw),0,g.cols);
 const r0=clamp(Math.floor(Math.min(...corners.map(p=>p.v))/ch),0,g.rows),r1=clamp(Math.ceil(Math.max(...corners.map(p=>p.v))/ch),0,g.rows);
 const numbersVisible=!n.child&&!!d&&!d.sample&&cellWidth>=44&&cellHeight>=22;
 n.grid={rows:g.rows,cols:g.cols,numbersVisible};
 if(!n.child&&cellWidth>=1.5&&cellHeight>=1.5){ctx.strokeStyle='#6b82734a';ctx.lineWidth=.5/S.scale;ctx.beginPath();for(let c=c0;c<=c1;c++){ctx.moveTo(c*cw,0);ctx.lineTo(c*cw,n.h);}for(let r=r0;r<=r1;r++){ctx.moveTo(0,r*ch);ctx.lineTo(n.w,r*ch);}ctx.stroke();}
 if(numbersVisible){for(let r=r0;r<r1;r++)for(let c=c0;c<c1;c++){const [y,x]=address(n,r,c);let val=d.data[(d.rgb?clamp(n.channel??S.c,0,2)*d.h*d.w:0)+y*d.w+x];if(selected&&S.edit?.moved&&y===S.y&&x===S.x)val=Number($('value').value);if(displayValue(val).length*6.1>cellWidth-8)continue;const u=active.u+(c+.5)*cw,v=active.v+(r+.5)*ch;numberLabels.push({x:n.x+.72*u,y:n.y-.38*u+v,text:displayValue(val)});}}
 if(selected){const row=g.transpose?S.x:S.y,col=g.transpose?S.y:S.x;if(!n.child&&cellWidth>=3){ctx.strokeStyle='#a86645';ctx.lineWidth=1.3/S.scale;ctx.strokeRect(col*cw,row*ch,cw,ch);}ctx.strokeStyle='#a86645';ctx.lineWidth=1.2/S.scale;ctx.strokeRect(0,0,n.w,n.h);cv.dataset.sectionRows=g.rows;cv.dataset.sectionColumns=g.cols;cv.dataset.numericCellsVisible=String(numbersVisible);}
 ctx.restore();
 const dpr=devicePixelRatio||1;ctx.save();ctx.setTransform(dpr,0,0,dpr,0,0);for(const label of numberLabels){const [x,y]=screenPoint(label.x,label.y);ctx.fillStyle='#334a3e';ctx.font=`10px ${MONO}`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.save();ctx.translate(x,y);ctx.transform(1,-.38/.72,0,1,0,0);ctx.fillText(label.text,0,0);ctx.restore();}ctx.restore();
}
function expansionCount(n){if(n.structural)return 1;return n.type==='input'||n.type==='output'||(n.type==='weight'&&n.shape.length<2)?1:dimensions(n)[0];}
function ghostFace(n){const b=nodeBounds(n);if(S.ox+b.right*S.scale<0||S.ox+b.left*S.scale>S.width||S.oy+b.bottom*S.scale<0||S.oy+b.top*S.scale>S.height)return;const g=depthGeometry(n);ctx.save();ctx.translate(n.x,n.y);ctx.transform(.72,-.38,0,1,0,0);ctx.strokeStyle='#81998a46';ctx.lineWidth=.6/S.scale;ctx.strokeRect(0,0,n.w,n.h);ctx.strokeStyle='#81998a24';ctx.strokeRect(g.du,g.dv,n.w,n.h);for(const [u,v] of [[0,0],[n.w,0],[n.w,n.h],[0,n.h]]){ctx.beginPath();ctx.moveTo(u,v);ctx.lineTo(u+g.du,v+g.dv);ctx.stroke();}ctx.restore();}
function terminal(n,right=true){return [n.x+(right?.72*n.w:0),n.y-(right?.38*n.w:0)+n.h/2];}
function faceCenter(n){return [n.x+.36*n.w,n.y-.19*n.w+n.h/2];}
function flowLink(a,b,highlight=false){const color=highlight?'#a86645':'#8f9e94',width=highlight?1.5:.65;line([a,b],color,width);const angle=Math.atan2(b[1]-a[1],b[0]-a[0]),l=(highlight?5:3)/S.scale;line([[b[0]-l*Math.cos(angle-.4),b[1]-l*Math.sin(angle-.4)],b,[b[0]-l*Math.cos(angle+.4),b[1]-l*Math.sin(angle+.4)]],color,width);}
function parameterLinks(stages){
 const routes=[];
 for(let i=0;i<stages.length;i++){
  const stage=stages[i],weights=S.nodes.filter(n=>n.type==='weight'&&n.id.startsWith(stage.id+'.')&&n.id.endsWith('.weight'));
  for(const weight of weights){
   const prefix=weight.id.slice(0,-7),bias=S.nodes.find(n=>n.id===prefix+'.bias'),child=stage.info.children.findIndex(n=>n.name===prefix),output=S.nodes.find(n=>n.id===prefix&&n.type==='activation')||stage,input=child>0?S.nodes.find(n=>n.id===stage.info.children[child-1].name):stages[i-1]||S.nodes[15];
   const highlight=!!S.selected&&[input,weight,bias,output].some(n=>n&&sameGroup(n,S.selected));
   const edges=[[terminal(input),terminal(weight,false)]];
   if(bias)edges.push([faceCenter(weight),faceCenter(bias)],[terminal(bias),terminal(output,false)]);
   else edges.push([terminal(weight),terminal(output,false)]);
   routes.push({edges,highlight});
  }
 }
 // Draw selected computation paths last so crossings remain easy to follow.
 for(const highlight of [false,true])for(const route of routes)if(route.highlight===highlight)for(const [a,b] of route.edges)flowLink(a,b,highlight);
 cv.dataset.highlightedFlowLinks=routes.filter(r=>r.highlight).reduce((sum,r)=>sum+r.edges.length,0);
}
function line(points,color='#8d999b',width=.65){ctx.strokeStyle=color;ctx.lineWidth=width/S.scale;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();}
function screenPoint(x,y){return [S.ox+x*S.scale,S.oy+y*S.scale];}
function screenLine(points,color='#a0aaa5',width=.65){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();}
function caption(text,x,y,{size=11,align='left',color='#425650',target=null,dy=0,background=false,angle=0}={}){const [sx,sy0]=screenPoint(x,y),sy=sy0+dy;size*=.72+.28*clamp((S.scale/minimumScale()-1)/3,0,1);ctx.font=`${size}px ${MONO}`;const width=ctx.measureText(text).width,left=align==='center'?-width/2:align==='right'?-width:0;if(sx+left>S.width+40||sx+left+width< -40||sy< -60||sy>S.height+60)return;ctx.save();ctx.translate(sx,sy);ctx.transform(1,Math.tan(angle),0,1,0,0);ctx.font=`${size}px ${MONO}`;ctx.textAlign=align;ctx.textBaseline='middle';if(background){ctx.fillStyle='#fafaf7';ctx.fillRect(left-6,-9,width+12,18);}ctx.fillStyle=color;ctx.fillText(text,0,0);ctx.restore();}
function annotate(stages){
 S.links=[];
 caption('01 / INPUT LATENT',75,-80,{size:12,target:'input',angle:AXIS_ANGLE});
 caption('z ['+S.shape.join(', ')+']',75,-80,{dy:21,size:10,target:'input',angle:AXIS_ANGLE});
 caption('02 / LEARNED WEIGHTS',START_X,-250,{size:12,target:'weights',angle:AXIS_ANGLE});
 caption('64 parameter tensors',START_X,-250,{dy:21,size:10,target:'weights',color:'#718079',angle:AXIS_ANGLE});
 const output=S.nodes.find(n=>n.type==='output');
 const groups=[[0,'STEM'],[3,'MEMORY 01'],[6,'EXPAND 01'],[9,'MEMORY 02'],[12,'EXPAND 02'],[15,'MEMORY 03'],[18,'EXPAND 03'],[22,'TO RGB']];
 for(const [i,text] of groups){const n=stages[i],[sx,sy]=screenPoint(n.x+.36*n.w,n.y+n.h+16),y=sy+26;caption(text,n.x+.36*n.w,n.y+n.h+16,{size:10,align:'center',dy:26,target:n,angle:AXIS_ANGLE});}
 for(const i of [1,13,19]){const n=stages[i];caption('['+n.shape.join(', ')+']',n.x+.36*n.w,n.y+n.h+16,{size:10,align:'center',dy:48,target:n,angle:AXIS_ANGLE});}
 const railEnd=stages.at(-1).x+stages.at(-1).w*.72+30,ax1=screenPoint(START_X,S.railBase),ax2=screenPoint(railEnd,axisY(railEnd,S.railBase));screenLine([ax1,ax2]);const nx=-Math.sin(AXIS_ANGLE),ny=Math.cos(AXIS_ANGLE);
 caption('23 OUTER OPERATIONS',(START_X+railEnd)/2,axisY((START_X+railEnd)/2,S.railBase),{align:'center',size:11,background:true,angle:AXIS_ANGLE});
 for(const [i,n] of stages.entries()){const tickX=n.x+.36*n.w,[x,y]=screenPoint(tickX,axisY(tickX,S.railBase));caption(String(i).padStart(2,'0'),tickX-24*Math.sin(AXIS_ANGLE)/S.scale,axisY(tickX,S.railBase)+24*Math.cos(AXIS_ANGLE)/S.scale,{size:9,align:'center',color:'#7e8b85',target:n,angle:AXIS_ANGLE});}
 // Compact overview notes occupy the bottom margin, clear of the tensors.
 {
 ctx.save();ctx.translate(24,S.height-100);ctx.scale(.8,.8);const sx=0,sy=0;
 ctx.textAlign='left';ctx.fillStyle='#425650';ctx.font=`11px ${MONO}`;
 ctx.fillText('TENSOR SHAPES',sx,sy);ctx.fillText('ARCHITECTURE',sx+275,sy);
 ctx.font=`10px ${MONO}`;ctx.fillStyle='#62716b';
 const entries=[['Input',S.shape],['Stem',stages[1].shape],['Temporal ×2',stages[13].shape],['Temporal ×4',stages[19].shape],['RGB',S.nodes.find(n=>n.type==='output').shape]];
 entries.forEach(([name,shape],i)=>{const yy=sy+23+i*19;ctx.fillText(name,sx,yy);ctx.fillText('['+shape.join(', ')+']',sx+85,yy);});
 ['23 outer operations / 64 parameter tensors','3 memory groups / 9 residual blocks',`Input section: x → ${S.shape[3]}, y ↓ ${S.shape[2]}`].forEach((text,i)=>ctx.fillText(text,sx+275,sy+23+i*19));
 ctx.restore();}
 const n=S.selected;if(n?.type==='weight'){const o=sectionOrigin(n);caption('X · W + b → Y',n.x+.72*o.u,n.y-.38*o.u+o.v+n.h,{dy:-12,size:10,background:true,angle:AXIS_ANGLE});}

}
function render(){const r=cv.getBoundingClientRect(),dpr=devicePixelRatio||1;if(r.width!==S.width||r.height!==S.height){S.width=r.width;S.height=r.height;cv.width=r.width*dpr;cv.height=r.height*dpr;S.dirty=true;constrainPan();}if(!S.dirty)return;S.dirty=false;S.numberLabels=[];delete cv.dataset.sectionRows;delete cv.dataset.sectionColumns;delete cv.dataset.numericCellsVisible;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,S.width,S.height);ctx.save();ctx.translate(S.ox,S.oy);ctx.scale(S.scale,S.scale);
 const stages=S.nodes.filter(n=>n.type==='activation'&&!n.child),output=S.nodes.find(n=>n.type==='output'),end=output.x+.72*output.w;
 line([[515,805],[START_X-45,axisY(START_X-45,START_Y+S.shape[3]*8*.75+45)],[end,axisY(end,START_Y+S.shape[3]*8*.75+45)]],'#70877c');
 line([[START_X-45,axisY(START_X-45,START_Y-100)],[end,axisY(end,START_Y-100)]],'#bbc5be');
 drawVideoConnection(output);
 parameterLinks(stages);
 if(S.selected){const root=rootOf(S.selected),C=expansionCount(root);for(let c=1;c<C;c++)ghostFace(channelNode(root,c));}
 S.drawn=[];for(const n of S.nodes.filter(n=>!n.structural).sort((a,b)=>a.x-b.x||a.y-b.y)){face(n);S.drawn.push(n);if(S.scale>1.4&&n.type==='weight'&&!S.weights.has(n.id)&&!pendingWeights.has(n.id)){const b=nodeBounds(n);if(S.ox+b.right*S.scale>=0&&S.ox+b.left*S.scale<=S.width&&S.oy+b.bottom*S.scale>=0&&S.oy+b.top*S.scale<=S.height){pendingWeights.add(n.id);binary('/api/weight?name='+encodeURIComponent(n.id)).then(b=>{S.weights.set(n.id,b.data);S.dirty=true;}).catch(fail).finally(()=>pendingWeights.delete(n.id));}}}
 if(S.selected){const root=rootOf(S.selected),C=expansionCount(root);cv.dataset.expandedChannels=C;cv.dataset.expandedRows=Math.min(8,C);cv.dataset.expandedColumns=Math.ceil(C/Math.min(8,C));cv.dataset.expandedSlice=S.t;cv.dataset.ghostChannels=C-1;}else{delete cv.dataset.expandedChannels;delete cv.dataset.expandedRows;delete cv.dataset.expandedColumns;delete cv.dataset.expandedSlice;delete cv.dataset.ghostChannels;}if(S.job?.state==='running'){const n=stages.find(n=>n.id===S.job.current_layer);if(n){ctx.fillStyle='#a96645';ctx.beginPath();ctx.arc(n.x-12,n.y+n.h+8,3/S.scale,0,Math.PI*2);ctx.fill();}}
 ctx.restore();positionPresets();annotate(stages);drawLocal();positionTools();$('zoom').textContent=Math.round(S.scale*100)+'%';
}
function minimumScale(){return Math.min(S.width/S.worldWidth,Math.max(260,S.height-170)/S.worldHeight)*.96;}
function fit(){S.zoomLevel=0;S.focusNode=null;cv.dataset.zoomLevel='0';S.scale=minimumScale();S.ox=(S.width-S.worldWidth*S.scale)/2;S.oy=130;S.dirty=true;}
function zoom(factor,x=S.width/2,y=S.height/2){hideLens();const scale=clamp(S.scale*factor,minimumScale(),2048);S.ox=x-(x-S.ox)*scale/S.scale;S.oy=y-(y-S.oy)*scale/S.scale;S.scale=scale;constrainPan();S.dirty=true;}
function point(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left-S.ox)/S.scale,(e.clientY-r.top-S.oy)/S.scale];}
function hit(e){const [x,y]=point(e);const nodes=[...(S.drawn||[])].reverse();return nodes.find(n=>{const b=nodeBounds(n);if(x<b.left||x>b.right||y<b.top||y>b.bottom)return false;const g=depthGeometry(n),p=localPoint(x,y,n);if(p.u>=0&&p.u<=n.w&&p.v>=0&&p.v<=n.h)return true;for(let t=g.count-1;t>=0;t--){const q=localPoint(x,y,n,t);if(q.u>=0&&q.u<=n.w&&q.v>=0&&q.v<=n.h)return true;}return false;});}
function cell(e,n){if(!n.grid)return;const [x,y]=point(e),{u,v}=localPoint(x,y,n),g=n.grid;if(u<0||v<0||u>n.w||v>n.h)return false;const row=clamp(Math.floor(v/n.h*g.rows),0,g.rows-1),col=clamp(Math.floor(u/n.w*g.cols),0,g.cols-1);[S.y,S.x]=address(n,row,col);return true;}
function nodeBounds(n){const g=depthGeometry(n);return {left:n.x+.72*g.du,right:n.x+.72*n.w,top:n.y-.38*(n.w+g.du)+g.dv,bottom:n.y+n.h};}
function frameTarget(target){const nodes=target==='input'?S.nodes.filter(n=>n.type==='input'):target==='weights'?S.nodes.filter(n=>n.type==='weight'):[target];const bounds=nodes.map(nodeBounds),left=Math.min(...bounds.map(b=>b.left)),right=Math.max(...bounds.map(b=>b.right)),top=Math.min(...bounds.map(b=>b.top)),bottom=Math.max(...bounds.map(b=>b.bottom));const available=Math.max(260,S.width-500);S.scale=clamp(Math.min(2.6,(available-90)/(right-left),(S.height-170)/(bottom-top)),minimumScale(),2048);S.ox=available/2-(left+right)/2*S.scale;S.oy=(S.height-30)/2-(top+bottom)/2*S.scale;S.dirty=true;}
// The camera keeps at least the bounded atlas within reach at every scale.
function constrainPan(){if(!S.width||!S.height)return;let left=0,right=S.worldWidth,top=0,bottom=S.worldHeight;const mx=S.width*.12,my=S.height*.12;function limit(v,size,min,max,margin,center){return (max-min)*S.scale<=size?clamp(v,center-margin,center+margin):clamp(v,size-margin-max*S.scale,margin-min*S.scale);}S.ox=limit(S.ox,S.width,left,right,mx,(S.width-S.worldWidth*S.scale)/2);S.oy=limit(S.oy,S.height,top,bottom,my,130);}
const VIDEO_DEPTH={x:150,y:-150*.38/.72};
function videoRoute(){
 const p=S.videoPanel,output=S.nodes.find(n=>n.type==='output'),start=[output.x+.72*output.w,output.y-.38*output.w],target=[p.x+p.w,p.y+p.h+p.w*AXIS_SLOPE];
 const run=(start[1]-target[1]-AXIS_SLOPE*(start[0]-target[0]))/(.38/.72+AXIS_SLOPE);
 return {start,target,turn:[start[0]+run,start[1]-run*.38/.72]};
}
function drawVideoConnection(output){
 const p=S.videoPanel,{start,turn,target}=videoRoute();line([start,turn,target],'#70877c');
 for(let j=4;j>=0;j--){const f=j/4;ctx.save();ctx.translate(p.x+VIDEO_DEPTH.x*f,p.y+VIDEO_DEPTH.y*f);ctx.transform(1,AXIS_SLOPE,0,1,0,0);if(j===4){ctx.fillStyle='#eff2ed';ctx.fillRect(0,0,p.w,p.h);}ctx.strokeStyle=j===0?'#82958a':'#a8b6aa';ctx.lineWidth=.65/S.scale;ctx.strokeRect(0,0,p.w,p.h);ctx.restore();}
 for(const [u,v] of [[0,0],[p.w,0],[0,p.h],[p.w,p.h]])line([[p.x+u,p.y+v+u*AXIS_SLOPE],[p.x+u+VIDEO_DEPTH.x,p.y+v+u*AXIS_SLOPE+VIDEO_DEPTH.y]],'#a8b6aa');
}
function positionVideo(){
 const video=$('video'),p=S.videoPanel;video.hidden=!S.result||!!S.videoBlank;$('video-actions').hidden=$('video-time').hidden=false;if(!p)return;
 const frames=S.result?.shape[1]||1,frame=S.videoBlank?0:(S.videoScrubFrame??Math.min(frames-1,Math.floor(video.currentTime*(S.result?.fps||16)))),f=frames>1?frame/(frames-1):0;
 const [x,y]=screenPoint(p.x+VIDEO_DEPTH.x*f,p.y+VIDEO_DEPTH.y*f);video.style.width=p.w*S.scale+'px';video.style.height=p.h*S.scale+'px';video.style.transform=`matrix(1,${AXIS_SLOPE},0,1,${x},${y})`;video.style.opacity=S.selected?'.18':'1';video.style.borderColor=S.videoSelected?'#a86645':'#82958a';
 const {target,turn}=videoRoute(),[ax,ay]=screenPoint((target[0]+turn[0])/2,(target[1]+turn[1])/2);
 const actions=$('video-actions');actions.style.left=ax+'px';actions.style.top=(ay-36)+'px';actions.style.transform=`translateX(-50%) skewY(${Math.atan(AXIS_SLOPE)}rad)`;
 const time=$('video-time');time.textContent='t = '+frame+' / '+(frames-1);
 const font=(10*(.72+.28*clamp((S.scale/minimumScale()-1)/3,0,1)))+'px';time.style.fontSize=font;actions.style.fontSize=(parseFloat(font)*1.2)+'px';
 const edge=screenPoint(p.x+p.w+VIDEO_DEPTH.x,p.y)[0],cx=Math.max(ax+18,edge+actions.offsetWidth/2+12);actions.style.left=cx+'px';actions.style.top=(ay+(cx-ax)*AXIS_SLOPE-36)+'px';
 $('video-play').disabled=!S.result;$('video-reset').disabled=!S.result;$('video-download').setAttribute('aria-disabled',String(!S.result));
 $('video-play').textContent=video.paused?'PLAY':'PAUSE';$('video-download').href=S.result?.video||'';$('video-download').download='wan-decoded-video.mp4';
}
function positionTools(){ $('tools').hidden=!S.selected||!!S.selected?.structural;positionVideo(); }

function drawLocal(){
 const n=S.selected;$('local-view').hidden=!n;if(!n){S.localGrid=null;return;}
 const lc=$('local-canvas'),r={width:lc.clientWidth,height:lc.clientHeight},dp=devicePixelRatio||1;
 if(lc.width!==Math.round(r.width*dp)||lc.height!==Math.round(r.height*dp)){lc.width=Math.round(r.width*dp);lc.height=Math.round(r.height*dp);}
 const q=lc.getContext('2d');q.setTransform(dp,0,0,dp,0,0);q.clearRect(0,0,r.width,r.height);
 const d=dataFor(n),g=layout(n),[C,T,H,W]=dimensions(n),t=sliceIndex(n),c=n.type==='input'?n.c:n===S.selected?S.c:0;
 $('local-name').textContent=n.id+' ['+n.shape.join(', ')+']';
 q.font=`10px ${MONO}`;q.fillStyle='#52675d';q.fillText((n.type==='weight'?'i':'t')+' = '+t+' / '+(T-1),10,17);
 const w=Math.min(78,112*g.cols/g.rows),h=Math.min(112,78*g.rows/g.cols),bx=22,by=43;
 // A full, uncropped slice sits beside an explicitly indexed numeric neighborhood.
 for(let j=T-1;j>=0;j--){const f=T>1?j/(T-1):0;q.strokeStyle='#9bacaa40';q.strokeRect(bx-f*14,by-f*10.5,w,h);}
 const displacement=T>1?t/(T-1):0,px=bx-displacement*14,py=by-displacement*10.5;
 q.fillStyle='#e1e9e2';q.fillRect(px,py,w,h);if(d){q.imageSmoothingEnabled=false;q.drawImage(cachedTexture(n,d,g.transpose),px,py,w,h);}q.strokeStyle='#79938b';q.strokeRect(px,py,w,h);
 const row=n===S.selected?(g.transpose?S.x:S.y):0,col=n===S.selected?(g.transpose?S.y:S.x):0;
 const rows=Math.min(5,g.rows),cols=Math.min(5,g.cols),r0=clamp(row-2,0,g.rows-rows),c0=clamp(col-2,0,g.cols-cols);
 const left=125,top=32,cw=Math.min(53,(r.width-left-8)/cols),ch=24;
 const rx=px+c0/g.cols*w,ry=py+r0/g.rows*h,rw=cols/g.cols*w,rh=rows/g.rows*h;
 q.fillStyle='#de512c38';q.fillRect(rx,ry,rw,rh);
 q.strokeStyle='#ffffff';q.lineWidth=4;q.strokeRect(rx,ry,rw,rh);
 q.strokeStyle='#c43c20';q.lineWidth=2;q.strokeRect(rx,ry,rw,rh);
 // Locator ticks keep a tiny crop legible without enlarging its actual area.
 q.beginPath();q.moveTo(rx-5,ry+rh/2);q.lineTo(rx-2,ry+rh/2);q.moveTo(rx+rw+2,ry+rh/2);q.lineTo(rx+rw+5,ry+rh/2);q.moveTo(rx+rw/2,ry-5);q.lineTo(rx+rw/2,ry-2);q.moveTo(rx+rw/2,ry+rh+2);q.lineTo(rx+rw/2,ry+rh+5);q.stroke();q.lineWidth=.65;q.strokeStyle='#ad8068';
 q.setLineDash([3,3]);q.beginPath();q.moveTo(px+w,py);q.lineTo(left,top);q.moveTo(px+w,py+h);q.lineTo(left,top+rows*ch);q.stroke();q.setLineDash([]);
 S.localGrid={n,left,top,cw,ch,rows,cols,r0,c0,px,py,w,h};
 q.textAlign='center';q.fillStyle='#6d7e74';q.fillText(g.transpose?'y →':'x →',left+(cols*cw)/2,29);
 for(let a=0;a<rows;a++)for(let b=0;b<cols;b++){
  const [y,x]=address(n,r0+a,c0+b),v=d&&!d.sample?d.data[(d.rgb?c*H*W:0)+y*W+x]:undefined;
  q.fillStyle=v===undefined?'#f3f5f0':'rgb('+rgb(v).join(',')+')';q.fillRect(left+b*cw,top+a*ch,cw,ch);
  q.strokeStyle='#aab5aa';q.lineWidth=.5;q.strokeRect(left+b*cw,top+a*ch,cw,ch);
  q.fillStyle='#344c40';const label=v===undefined?'—':displayValue(v);q.font=`${Math.min(10,(cw-4)/(label.length*.61))}px ${MONO}`;q.fillText(label,left+(b+.5)*cw,top+a*ch+18);q.font=`10px ${MONO}`;
  if(r0+a===row&&c0+b===col){q.strokeStyle='#c43c20';q.lineWidth=2;q.strokeRect(left+b*cw,top+a*ch,cw,ch);}
 }
 const [yy,xx]=address(n,row,col);q.textAlign='left';q.fillStyle='#52675d';q.fillText('['+c+', '+t+', '+yy+', '+xx+']',left,top+rows*ch+20);
 q.fillText(d&&!d.sample?(n.type==='input'?'EDITABLE':'READ ONLY'):'Compute to inspect values',left,top+rows*ch+37);
 const corner=$('local-view').getBoundingClientRect(),canvasTop=cv.getBoundingClientRect().top;
 const rect={left:corner.left,right:corner.right,top:corner.top-canvasTop,bottom:corner.bottom-canvasTop},o=sectionOrigin(n);
 const candidates=[[0,0],[n.w,0],[n.w,n.h],[0,n.h]].map(([u,v])=>{const [sx,sy]=screenPoint(n.x+.72*(u+o.u),n.y-.38*(u+o.u)+v+o.v),ex=clamp(sx,rect.left,rect.right),ey=clamp(sy,rect.top,rect.bottom);return {sx,sy,ex,ey,d:Math.hypot(sx-ex,sy-ey)};}).filter(p=>p.sx>=0&&p.sx<=S.width&&p.sy>=0&&p.sy<=S.height).sort((a,b)=>a.d-b.d);
 const lead=candidates[0];if(lead&&lead.d>1)screenLine([[lead.sx,lead.sy],[lead.ex,lead.ey]],'#8b9e91',.7);
}
function cycleZoom(e){
 hideLens();
 if(S.zoomLevel===2){cancelSliceLoad();S.selected=null;controls();fit();return;}
 const [wx,wy]=point(e),r=cv.getBoundingClientRect();
 let n=hit(e)||S.focusNode||S.nodes.reduce((best,v)=>{const b=nodeBounds(v),distance=Math.hypot((b.left+b.right)/2-wx,(b.top+b.bottom)/2-wy);return !best||distance<best.distance?{n:v,distance}:best;},null)?.n;
 n=rootOf(n);S.focusNode=n;S.zoomLevel++;
 if(n){if(S.selected!==n)select(n).catch(fail);
  if(S.zoomLevel===1){frameTarget(n);}
  else{const g=layout(n),o=sectionOrigin(n),row=g.transpose?S.x:S.y,col=g.transpose?S.y:S.x;
   S.scale=clamp(Math.max(S.scale*1.5,52/(n.w/g.cols*.72),30/(n.h/g.rows)),minimumScale(),2048);
   const u=o.u+(col+.5)*n.w/g.cols,v=o.v+(row+.5)*n.h/g.rows;
   S.ox=S.width*.3-(n.x+.72*u)*S.scale;S.oy=S.height*.5-(n.y-.38*u+v)*S.scale;
  }
 }else zoom(3,e.clientX-r.left,e.clientY-r.top);
 constrainPan();S.dirty=true;cv.dataset.zoomLevel=String(S.zoomLevel);
}
function controls(){let n=S.selected;if(n?.structural){$('tools').hidden=true;$('capture').hidden=true;$('value-label').hidden=true;S.dirty=true;return;}if(!n){$('tools').hidden=true;$('selection').textContent='';S.dirty=true;return;}const [C,T,H,W]=dimensions(n);S.c=Math.trunc(clamp(S.c,0,C-1));if(n.type!=='input'){n=rootOf(n);S.selected=n;}S.t=Math.trunc(clamp(S.t,0,T-1));S.y=Math.trunc(clamp(S.y,0,H-1));S.x=Math.trunc(clamp(S.x,0,W-1));$('identity').textContent=n.id+' ['+n.shape.join(',')+']'+(n.type==='weight'?' · readonly':'')+' · '+(n.type==='weight'?'i':'t')+'='+S.t+'/'+(T-1)+' · '+T+' slices';$('selection').textContent='';$('channel-label').hidden=n.type==='input'||n.type==='output';$('depth-label').textContent=n.type==='weight'?'i':'t';for(const [id,val,max] of [['channel',S.c,C-1],['depth',S.t,T-1],['row',S.y,H-1],['col',S.x,W-1]]){$(id).value=val;$(id).max=max;}$('capture').hidden=n.type!=='activation';$('save').hidden=n.type!=='input';const d=dataFor(n);$('value-label').hidden=!d||!!d.sample;$('value').readOnly=n.type!=='input';if(d&&!d.sample)$('value').value=decimalValue(d.data[(d.rgb?S.c*d.h*d.w:0)+S.y*d.w+S.x]);S.dirty=true;}
async function select(n){S.videoSelected=false;cancelSliceLoad();S.selected=rootOf(n);S.c=n.channel??0;S.t=clamp(S.t,0,dimensions(n)[1]-1);S.x=S.y=0;controls();const request=++S.request;if(n.type==='weight'&&!S.weights.has(n.id)){const b=await binary('/api/weight?name='+encodeURIComponent(n.id));S.weights.set(n.id,b.data);if(request===S.request)controls();}if(n.type==='activation'){await loadActivation(n);}if(n.type==='output')await loadRGB();}
async function loadDecodedPlane(n,t=0,signal){const m=S.result;if(!m?.activations?.[n.id]||t>=m.activations[n.id].shape[0])return;const key=n.id+':'+t;if(S.decodePlanes?.get(key)?.id===m.id)return;cv.dataset.sliceReads=String(Number(cv.dataset.sliceReads||0)+1);const b=await binary('/api/decode-activation?id='+m.id+'&layer='+encodeURIComponent(n.id)+'&time='+t,signal);if(S.result?.id!==m.id||signal?.aborted)return;S.decodePlanes ||= new Map();S.decodePlanes.delete(key);S.decodePlanes.set(key,{...b,id:m.id,revision:m.revision});while(S.decodePlanes.size>120){const oldest=[...S.decodePlanes.keys()].find(k=>!k.endsWith(':0'));if(!oldest)break;S.decodePlanes.delete(oldest);}S.dirty=true;}
async function loadActivation(n,signal){const requestedTime=S.t,requestedChannel=S.c;if(S.c===0&&S.result?.activations?.[n.id]&&S.t<S.result.activations[n.id].shape[0]){await loadDecodedPlane(n,requestedTime,signal);if(S.selected===n&&S.t===requestedTime&&S.c===requestedChannel&&!signal?.aborted)controls();return;}const a=S.activations.get(n.id);if(!a||a.t!==S.t||a.c===S.c)return;const c=S.c;try{const b=await binary('/api/activation?id='+a.id+'&c='+c,signal);if(signal?.aborted)return;const plane={...a,...b,c};S.activations.set(n.id,plane);S.activationPlanes ||= new Map();S.activationPlanes.set(n.id+':'+plane.c+':'+plane.t,plane);if(S.selected?.id===n.id&&S.c===c)controls();S.dirty=true;}catch(e){if(e.name==='AbortError')throw e;S.activations.delete(n.id);S.dirty=true;}}
async function save(v){const n=S.selected;if(!n||n.type!=='input'||!Number.isFinite(v))return;const addr=[n.c,S.t,S.y,S.x];const status=await api('/api/edit',{revision:S.revision,mode:'scalar',c:addr[0],t:addr[1],y:addr[2],x:addr[3],value:v});S.z[index(...addr)]=v;S.revision=status.revision;controls();}
const rgbFrames=new Map();
async function loadRGB(signal){if(!S.result)return;const result=S.result,request=S.rgbRequest=(S.rgbRequest||0)+1,frame=clamp(S.t,0,result.shape[1]-1),key=result.id+':'+frame;
 let data=rgbFrames.get(key);if(!data)data=await binary('/api/rgb?id='+result.id+'&frame='+frame,signal);
 if(signal?.aborted||S.result?.id!==result.id||request!==S.rgbRequest)return;
 rgbFrames.delete(key);rgbFrames.set(key,data);while(rgbFrames.size>8)rgbFrames.delete(rgbFrames.keys().next().value);
 S.rgb=data;S.rgbFrame=frame;S.dirty=true;if(S.selected?.type==='output')controls();}

async function poll(){if(S.switching)return;const epoch=S.request,status=await api('/api/status');if(S.switching||epoch!==S.request||status.revision!==S.revision)return;S.job=status.jobs.find(j=>j.state==='running');$('decode').disabled=$('capture').disabled=!!S.job;document.querySelectorAll('#latent-presets button').forEach(b=>b.disabled=!!S.job||!!S.switching);$('status').textContent=S.job?Math.round(S.job.progress*100)+'%':'';if(status.trace?.revision===S.revision&&!S.done.has(status.trace.id)){const m=status.trace;S.done.add(m.id);const c=S.selected?.id===m.layer?clamp(S.c,0,m.shape[0]-1):0;const b=await binary('/api/activation?id='+m.id+'&c='+c);S.activations.set(m.layer,{...b,id:m.id,t:m.time,c,revision:m.revision});S.dirty=true;if(S.selected?.id===m.layer){controls();}}const last=status.results.filter(r=>r.revision===S.revision).at(-1);if(last&&last.id!==S.result?.id){S.result=last;S.videoBlank=false;S.decodePlanes=new Map();clearTextures();clearTimeout(videoSeekTimer);S.videoScrubFrame=null;$('video').src=last.video;await loadRGB();for(const n of S.nodes.filter(n=>n.type==='activation'))await loadDecodedPlane(n,0);controls();}const failed=status.jobs.at(-1);if(failed?.state==='error')fail(failed.error);if(S.job)S.dirty=true;}
cv.addEventListener('wheel',e=>{hideLens();e.preventDefault();if(S.edit)return;if(S.selected&&!e.ctrlKey&&!e.metaKey){S.t=clamp(S.t+Math.sign(e.deltaY),0,dimensions(S.selected)[1]-1);scheduleSliceLoad();}else{const r=cv.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.002),e.clientX-r.left,e.clientY-r.top);}},{passive:false});
cv.addEventListener('pointerdown',e=>{hideLens();if(e.button!==0)return;const n=hit(e);if(n&&n===S.selected&&n.grid?.numbersVisible&&cell(e,n)){controls();if(n.type==='input'){S.edit={y:e.clientY,value:S.z[index(n.c,S.t,S.y,S.x)],moved:false};cv.setPointerCapture(e.pointerId);return;}}S.drag={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY,moved:false};cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(S.edit){const d=S.edit.y-e.clientY;S.edit.moved ||= Math.abs(d)>3;if(S.edit.moved){$('value').value=S.edit.value+d*(e.shiftKey?.001:.01);S.dirty=true;}return;}if(!S.drag){queueHover(e);return;}S.drag.moved ||= Math.abs(e.clientX-S.drag.sx)+Math.abs(e.clientY-S.drag.sy)>4;S.ox+=e.clientX-S.drag.x;S.oy+=e.clientY-S.drag.y;S.drag.x=e.clientX;S.drag.y=e.clientY;constrainPan();S.dirty=true;});
cv.addEventListener('pointerup',e=>{if(S.edit){const edit=S.edit;S.edit=null;if(edit.moved)save(Number($('value').value)).catch(fail);return;}const moved=S.drag?.moved;S.drag=null;if(!moved){const n=hit(e);if(n){if(n!==S.selected)select(n).catch(fail);else{cell(e,n);controls();}}else{cancelSliceLoad();S.selected=null;controls();}}});cv.addEventListener('dblclick',e=>{e.preventDefault();cycleZoom(e);});cv.addEventListener('pointercancel',()=>{S.drag=S.edit=null;controls();});
$('local-canvas').addEventListener('click',async e=>{try{
 const g=S.localGrid;if(!g)return;const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)*e.currentTarget.clientWidth/r.width,y=(e.clientY-r.top)*e.currentTarget.clientHeight/r.height;
 if(S.selected!==g.n)await select(g.n);
 if(x>=g.left&&x<g.left+g.cols*g.cw&&y>=g.top&&y<g.top+g.rows*g.ch){[S.y,S.x]=address(g.n,g.r0+Math.floor((y-g.top)/g.ch),g.c0+Math.floor((x-g.left)/g.cw));}
 else if(x>=g.px&&x<g.px+g.w&&y>=g.py&&y<g.py+g.h){const shape=layout(g.n);[S.y,S.x]=address(g.n,Math.floor((y-g.py)/g.h*shape.rows),Math.floor((x-g.px)/g.w*shape.cols));}
 controls();
}catch(e){fail(e);}});
$('local-canvas').addEventListener('wheel',e=>{e.preventDefault();const n=S.localGrid?.n;if(!n)return;if(S.selected!==n){select(n).catch(fail);return;}S.t=clamp(S.t+Math.sign(e.deltaY),0,dimensions(n)[1]-1);scheduleSliceLoad();},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){cancelSliceLoad();S.selected=null;controls();}});
for(const [id,key] of [['channel','c'],['depth','t'],['row','y'],['col','x']])$(id).addEventListener('input',()=>{S[key]=Number($(id).value);if(key==='c'||key==='t')scheduleSliceLoad();else controls();});
action('save',()=>save(Number($('value').value)));$('value').addEventListener('keydown',e=>{if(e.key==='Enter')save(Number($('value').value)).catch(fail);});
action('fit',()=>{cancelSliceLoad();S.selected=null;controls();fit();});for(const [id,f] of [['plus',1.5],['minus',1/1.5]])action(id,()=>{const n=S.selected;zoom(f,n?S.ox+n.x*S.scale:S.width/2,n?S.oy+n.y*S.scale:S.height/2);});
action('decode',async()=>{await api('/api/decode',{});await poll();});action('capture',async()=>{await api('/api/trace',{layer:S.selected.id,time:S.t});await poll();});
let videoSeekTimer=null;
$('video').addEventListener('click',()=>{cancelSliceLoad();S.selected=null;S.videoSelected=true;controls();positionVideo();});
$('video').addEventListener('wheel',e=>{e.preventDefault();if(e.ctrlKey||e.metaKey){const r=cv.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.002),e.clientX-r.left,e.clientY-r.top);return;}if(!S.videoSelected||!S.result)return;
 const video=$('video'),fps=S.result.fps,frame=S.videoScrubFrame??Math.floor(video.currentTime*fps);video.pause();S.videoScrubFrame=clamp(frame+Math.sign(e.deltaY||e.deltaX),0,S.result.shape[1]-1);clearTimeout(videoSeekTimer);positionVideo();
 videoSeekTimer=setTimeout(()=>{video.currentTime=S.videoScrubFrame/fps;},120);
},{passive:false});
$('video').addEventListener('seeked',()=>{const target=S.videoScrubFrame;if(target!==null&&target!==undefined&&Math.abs($('video').currentTime-target/(S.result?.fps||16))<.02)S.videoScrubFrame=null;positionVideo();});
action('video-reset',()=>{clearTimeout(videoSeekTimer);S.videoScrubFrame=null;S.videoSelected=false;S.videoBlank=true;const video=$('video');video.pause();video.currentTime=0;positionVideo();S.dirty=true;});
$('video-download').addEventListener('click',e=>{if(!S.result)e.preventDefault();});
action('video-play',async()=>{if(!S.result)return;S.videoBlank=false;clearTimeout(videoSeekTimer);const video=$('video');if(S.videoScrubFrame!=null)video.currentTime=S.videoScrubFrame/(S.result?.fps||16);S.videoScrubFrame=null;if(video.paused){if(video.ended)video.currentTime=0;await video.play();}else video.pause();positionVideo();});
for(const event of ['loadeddata','timeupdate','play','pause','ended'])$('video').addEventListener(event,positionVideo);
if('requestVideoFrameCallback' in $('video')){const update=()=>{positionVideo();$('video').requestVideoFrameCallback(update);};$('video').requestVideoFrameCallback(update);}
function positionPresets(){const el=$('latent-presets');el.style.left=Math.max(8,S.ox-38)+'px';el.style.top=(S.oy+210*S.scale)+'px';el.style.visibility=S.ox<0||S.ox>S.width?'hidden':'visible';}
async function initPresets(){const items=await api('/api/presets'),el=$('latent-presets');el.replaceChildren();for(const item of items){const b=document.createElement('button');b.dataset.label=item.label;b.dataset.id=item.id;b.title=item.label;b.setAttribute('aria-label',item.label);b.setAttribute('aria-pressed',String(item.id===S.preset));b.addEventListener('click',()=>selectPreset(item.id).catch(fail));el.append(b);}}
async function selectPreset(id){
 if(S.switching)return;S.switching=true;S.request++;cancelSliceLoad();hideLens();$('error').textContent='';
 document.querySelectorAll('#latent-presets button').forEach(b=>b.disabled=true);$('decode').disabled=true;
 try{
  const status=await api('/api/preset',{id});
  const [z,model]=await Promise.all([binary('/api/latent'),api('/api/model')]);
  const video=$('video');video.pause();video.removeAttribute('src');video.load();clearTimeout(videoSeekTimer);
  S.z=z.data;S.shape=z.shape;S.model=model;S.revision=status.revision;S.preset=status.preset;S.result=null;S.rgb=null;S.rgbFrame=0;S.rgbRequest=(S.rgbRequest||0)+1;
  S.selected=null;S.job=null;S.videoBlank=true;S.videoSelected=false;S.videoScrubFrame=null;S.c=S.t=S.y=S.x=0;
  S.activations.clear();S.activationPlanes?.clear();S.decodePlanes?.clear();S.done.clear();rgbFrames.clear();lensPatches.clear();clearTextures();
  build();controls();fit();positionVideo();
  document.querySelectorAll('#latent-presets button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.id===id)));
 }finally{S.switching=false;document.querySelectorAll('#latent-presets button').forEach(b=>b.disabled=false);$('decode').disabled=false;}
}
async function boot(){try{[S.model,S.samples]=await Promise.all([api('/api/model'),api('/api/weight-previews')]);const [z,status]=await Promise.all([binary('/api/latent'),api('/api/status')]);S.z=z.data;S.shape=z.shape;S.revision=status.revision;S.preset=status.preset;await initPresets();build();render();fit();await poll();setInterval(()=>poll().catch(fail),1000);function frame(){render();requestAnimationFrame(frame);}frame();}catch(e){fail(e);}}


// A separate overlay keeps mouse motion from redrawing the atlas or loading any data.
let hoverFrame=0,hoverEvent=null,lensTimer=0,lensRequest=null,lensKey=null;
const lensPatches=new Map();
const lens=$('magnifier'),lensToggle=$('lens-toggle');
try{lensToggle.checked=localStorage.getItem('wan-magnifier')!=='off';}catch{}
function hideLens(){lens.hidden=true;hoverEvent=null;clearTimeout(lensTimer);lensRequest?.abort();lensKey=null;}
function queueHover(e){hoverEvent={clientX:e.clientX,clientY:e.clientY};if(hoverFrame)return;hoverFrame=requestAnimationFrame(()=>{hoverFrame=0;const ev=hoverEvent;if(!ev||S.drag||S.edit)return;const n=hit(ev);cv.style.cursor=n?'crosshair':'grab';if(!lensToggle.checked||!n){lens.hidden=true;return;}paintLens(ev,n);});}
function paintLens(e,n){
 if(n.structural){lens.hidden=true;return;}
 const [wx,wy]=point(e),p=localPoint(wx,wy,n),g=layout(n),cellW=n.w/g.cols,cellH=n.h/g.rows;
 if(S.zoomLevel===2||(cellW*.72*S.scale>=44&&cellH*S.scale>=22)||p.u<0||p.v<0||p.u>n.w||p.v>n.h){lens.hidden=true;return;}
 const row=clamp(Math.floor(p.v/cellH),0,g.rows-1),col=clamp(Math.floor(p.u/cellW),0,g.cols-1),[y,x]=address(n,row,col),d=dataFor(n),c=n.type==='input'?n.c:(n.channel??(sameGroup(n,S.selected)?S.c:0)),t=sliceIndex(n);
 const rows=Math.min(3,g.rows),cols=Math.min(3,g.cols),r0=clamp(row-1,0,g.rows-rows),c0=clamp(col-1,0,g.cols-cols);
 // Magnify the same face projection uniformly: x is compressed by .72,
 // vertical edges remain upright, and the horizontal edges shear by -.38/.72.
 const factor=Math.max(52/(cellW*.72),32/cellH)*.7,cw=cellW*.72*factor,ch=cellH*factor,slope=-.38/.72,pad=2;
 const depths=[t-1,t,t+1].filter(v=>v>=0&&v<dimensions(n)[1]);
 // Depth follows the atlas axis. Separate layers enough to read exposed cells.
 const stepX=cw*.72,stepY=stepX*AXIS_SLOPE;
 const offsets=depths.map(v=>({t:v,x:-(v-t)*stepX,y:-(v-t)*stepY}));
 const minX=Math.min(...offsets.map(v=>v.x)),minY=Math.min(...offsets.map(v=>v.y));
 const W=Math.ceil(cols*cw+Math.max(...offsets.map(v=>v.x))-minX+pad*2),H=Math.ceil(rows*ch-slope*cols*cw+Math.max(...offsets.map(v=>v.y))-minY+pad*2),dp=devicePixelRatio||1;
 const fitScale=Math.sqrt(((cols*cw+pad*2)*(rows*ch-slope*cols*cw+pad*2))/(W*H)),displayW=W*fitScale,displayH=H*fitScale;
 if(lens.width!==W*dp||lens.height!==H*dp){lens.width=W*dp;lens.height=H*dp;}lens.style.width=displayW+'px';lens.style.height=displayH+'px';
 const key=[S.revision,S.result?.id,n.id,c,t,r0,c0].join(':');
 const patch=lensPatches.get(key);
 if(lensKey!==key){
  clearTimeout(lensTimer);lensRequest?.abort();lensKey=key;
  if(!patch)lensTimer=setTimeout(async()=>{
   const controller=new AbortController();lensRequest=controller;
   try{const params=new URLSearchParams({name:n.type==='output'?'rgb':n.id,c,t,y:r0,x:c0,revision:S.revision});
    const response=await fetch('/api/lens?'+params,{signal:controller.signal});if(!response.ok)return;
    const data=await response.json();if(controller.signal.aborted)return;
    lensPatches.set(key,data);while(lensPatches.size>96)lensPatches.delete(lensPatches.keys().next().value);
    if(lensKey===key&&hoverEvent&&!lens.hidden)paintLens(hoverEvent,n);
   }catch(error){if(error.name!=='AbortError')console.warn('Lens neighborhood unavailable',error);}
  },150);
 }
 const q=lens.getContext('2d');q.setTransform(dp,0,0,dp,0,0);q.clearRect(0,0,W,H);
 const ox=pad-minX,oy=pad-slope*cols*cw-minY;
 // Join corresponding corners to make the slice spacing visible as thickness.
 q.strokeStyle='#6b827360';q.lineWidth=.6;
 for(let i=1;i<offsets.length;i++)for(const [u,v] of [[0,0],[cols*cw,0],[0,rows*ch],[cols*cw,rows*ch]]){
  q.beginPath();q.moveTo(ox+offsets[i-1].x+u,oy+offsets[i-1].y+slope*u+v);q.lineTo(ox+offsets[i].x+u,oy+offsets[i].y+slope*u+v);q.stroke();
 }
 for(const layer of [...offsets].reverse()){
  const current=layer.t===t,plane=dataFor(n,layer.t),values=patch?.layers.find(a=>a.t===layer.t)?.values;
  q.save();q.translate(ox+layer.x,oy+layer.y);q.transform(1,slope,0,1,0,0);q.textAlign='center';q.textBaseline='middle';
  for(let r=0;r<rows;r++)for(let k=0;k<cols;k++){
   const [yy,xx]=address(n,r0+r,c0+k),v=values?.[r]?.[k]??plane?.data[yy*plane.w+xx],label=v===undefined?'—':displayValue(v);
   q.globalAlpha=current?.72:.16;q.fillStyle=v===undefined?'#dde8e2':'rgb('+rgb(v).join(',')+')';q.fillRect(k*cw,r*ch,cw,ch);
   q.globalAlpha=current?1:.45;q.strokeStyle='#6b82734a';q.lineWidth=.5;q.strokeRect(k*cw,r*ch,cw,ch);
   const focused=current&&r0+r===row&&c0+k===col;
   if(focused){q.strokeStyle='#a86645';q.lineWidth=1.5;q.strokeRect(k*cw+.75,r*ch+.75,cw-1.5,ch-1.5);}
   q.fillStyle=focused?'#a86645':'#334a3e';q.font=(focused?'600 ':'')+Math.min(11*.7,(cw-6)/(Math.max(1,label.length)*.61))+'px '+MONO;q.fillText(label,(k+.5)*cw,(r+.5)*ch);
  }
  q.restore();
 }
 lens.dataset.depths=JSON.stringify(depths);
 const val=d?.data[y*d.w+x];lens.dataset.tensor=n.id;lens.dataset.indices=JSON.stringify([c,t,y,x]);lens.dataset.value=val===undefined?'':String(val);lens.dataset.rows=rows;lens.dataset.cols=cols;
 lens.style.left=Math.max(8,Math.min(e.clientX+18,window.innerWidth-displayW-8))+'px';lens.style.top=Math.max(8,Math.min(e.clientY+18,window.innerHeight-displayH-8))+'px';lens.hidden=false;
}
lensToggle.addEventListener('change',()=>{hideLens();try{localStorage.setItem('wan-magnifier',lensToggle.checked?'on':'off');}catch{}});
cv.addEventListener('pointerup',e=>queueHover(e));
cv.addEventListener('pointerleave',hideLens);window.addEventListener('blur',hideLens);window.addEventListener('resize',hideLens);
boot();
