const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context={};vm.createContext(context);vm.runInContext(fs.readFileSync('web/full-model/wan-architecture.js','utf8'),context);
const a=context.makeWanArchitecture([16,21,60,104]);
assert.equal(a.nodes.filter(n=>/^wan\.text\.blocks\.\d+\.output$/.test(n.id)).length,24);
for(const n of a.nodes){assert(n.actualShape.every(v=>Number.isInteger(v)&&v>0));assert([n.x,n.y,n.w,n.h].every(Number.isFinite));}
assert.equal(a.groups.length,30);
for(let row=0;row<3;row++){
 const groups=a.groups.filter(g=>g.row===row);
 assert.equal(groups.length,10);
 assert.deepEqual(Array.from(groups,g=>g.col),Array.from({length:10},(_,i)=>row%2?9-i:i));
}
for(const n of a.nodes)assert(context.architectureVisible(n),'all internal tensors remain visible at overview');
for(let i=1;i<30;i++){
 const prev=a.groups[i-1],current=a.groups[i];
 if(prev.row===current.row)assert.equal(Math.sign(current.x-prev.x),current.row%2?-1:1);
 else assert(current.y>prev.y,'fold advances vertically');
}
assert.equal(a.N,32760);
assert.equal(a.byId.size,a.nodes.length,'unique graph node IDs');
for(const edge of a.edges){assert(a.byId.has(edge.from),edge.from);assert(a.byId.has(edge.to),edge.to);}
const shape=id=>Array.from(a.byId.get(id).actualShape);
for(let i=0;i<30;i++){
 const root=`wan.blocks.${i}`;
 assert.deepEqual(shape(root+'.output'),[1,32760,1536]);
 assert.deepEqual(shape(root+'.self_attn.scores'),[1,12,32760,32760]);
 assert.deepEqual(shape(root+'.cross_attn.scores'),[1,12,32760,512]);
 assert.deepEqual(shape(root+'.ffn.0.W'),[8960,1536]);
 assert.deepEqual(shape(root+'.ffn.2.W'),[1536,8960]);
 const g=a.groups[i];assert.equal(g.entry,i?`wan.blocks.${i-1}.output`:'wan.tokens');
 const connected=(from,to)=>a.edges.some(e=>e.from===from&&e.to===to);
 assert(connected(g.entry,root+'.residual1'));
 assert(connected(root+'.residual1',root+'.residual2'));
 assert(connected(root+'.residual2',root+'.output'));
}
assert.deepEqual(shape('wan.head.proj'),[1,32760,64]);
assert.deepEqual(shape('wan.clean'),[1,16,21,60,104]);
assert(a.edges.some(e=>e.from==='wan.scheduler'&&e.to==='wan.noise'));
console.log(`Verified ${a.groups.length} blocks, ${a.nodes.length} nodes, ${a.edges.length} connections and contraction dimensions`);
assert.equal(a.byId.get('wan.noise').role,'video-input');
assert.equal(a.byId.get('wan.prompt').role,'text-input');
const qw=a.byId.get('wan.blocks.0.self_attn.q.W');
assert.equal(qw.role,'parameter');
assert.deepEqual(Array.from(qw.shape),[1536,1,1,1536]);
assert(a.byId.get('wan.blocks.0.self_attn.scores').w > a.byId.get('wan.blocks.0.self_attn.q').w);
assert.equal(a.byId.get('wan.blocks.0.self_attn.scores').shape[0],12);
console.log('Verified input roles, fixed parameter rows and dimension-dependent geometry.');
