"""Layer Ablation: How many pretrained layers are needed for context generalization?

Tests 6 configs (0 to 10 standard layers). Each trains at 2048, evals at 2048 and 4096.
"""

import json, math, random, string, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 768; N_HEADS = 12; BLOCK_SIZE = 8; WINDOW_SIZE = 64
NUM_SACCADES = 5; CONTEXT_LENGTH = 2048
LR = 5e-4; BATCH_SIZE = 4; WEIGHT_DECAY = 0.01; GRAD_CLIP = 1.0
ENTROPY_BONUS = 0.01; WARMUP_STEPS = 50
SUPERVISED_WARMUP_STEPS = 200; SUPERVISED_WARMUP_WEIGHT = 2.0
GUMBEL_TEMP_START = 1.0; GUMBEL_TEMP_END = 0.1; GUMBEL_ANNEAL_STEPS = 500
NUM_TRAIN = 3000; NUM_EVAL = 200; WALL_CLOCK = 600

# Config A-F: (name, saccadic_layers)
ABLATION_CONFIGS = [
    ('A: 0std+12sacc', list(range(12))),
    ('B: 2std+10sacc', list(range(2, 12))),
    ('C: 4std+8sacc',  list(range(4, 12))),
    ('D: 6std+6sacc',  list(range(6, 12))),
    ('E: 8std+4sacc',  list(range(8, 12))),
    ('F: 10std+2sacc', list(range(10, 12))),
]

def log(msg): print(msg, file=sys.stderr, flush=True)
_tok = GPT2Tokenizer.from_pretrained('gpt2')
FILLER = [
    "The weather was pleasant and the sky was clear.",
    "Several researchers gathered to discuss the latest findings.",
    "The library contained thousands of books on various topics.",
    "Traffic moved slowly through the busy intersection.",
    "A gentle breeze rustled through the autumn leaves.",
    "The project deadline was approaching rapidly.",
    "Students worked diligently on their assignments.",
    "The old building stood at the corner of the street.",
    "New developments in technology continued to emerge.",
    "The garden was well maintained throughout the year.",
    "Several factors contributed to the overall outcome.",
    "The meeting was scheduled for early in the morning.",
    "A small group discussed various approaches to the problem.",
    "The document outlined the key objectives clearly.",
    "Regular maintenance ensured smooth operation of equipment.",
    "The analysis revealed several interesting patterns.",
    "Participants shared their experiences and insights.",
    "The report summarized findings from the past quarter.",
    "Careful planning led to a successful implementation.",
    "The results exceeded expectations for the quarter.",
]

class PasskeyDataset(Dataset):
    def __init__(self, n, ctx_len, seed=42):
        self.rng = random.Random(seed)
        self.filler_enc = [_tok.encode(" "+s) for s in FILLER]
        self.prompt = _tok.encode(" What is the secret number? The secret number is")
        self.samples = [self._make(i, ctx_len) for i in range(n)]
    def _make(self, idx, ctx_len):
        rng = random.Random(self.rng.randint(0,2**32)+idx)
        pk = ''.join(rng.choices(string.digits, k=5))
        pk_ids = _tok.encode(f" The secret number is {pk}.")
        ans = _tok.encode(" "+pk)
        budget = ctx_len-len(pk_ids)-len(self.prompt)-len(ans)
        fill = []
        while len(fill)<budget: fill.extend(rng.choice(self.filler_enc))
        fill = fill[:budget]; pos = rng.randint(0, len(fill))
        full = fill[:pos]+pk_ids+fill[pos:]+self.prompt+ans
        if len(full)>ctx_len: full=full[:ctx_len]
        elif len(full)<ctx_len: full+=[_tok.eos_token_id]*(ctx_len-len(full))
        return (torch.tensor(full[:ctx_len],dtype=torch.long),
                torch.tensor([int(d) for d in pk],dtype=torch.long),pk,pos)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids,dl,pk,pos = self.samples[i]
        return {'input_ids':ids,'digit_labels':dl,'passkey':pk,'passkey_position':pos}

def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) if k in ('input_ids','digit_labels')
            else [b[k] for b in batch] for k in batch[0]}

# ── Components ────────────────────────────────────────────────────────────────

class PeripheralEncoder(nn.Module):
    def __init__(self, dim, bs):
        super().__init__()
        self.bs=bs; self.w=nn.Linear(dim,1); self.proj=nn.Linear(dim*3,dim)
        self.norm=nn.LayerNorm(dim); self.pos=nn.Embedding(16384,dim)
    def forward(self, x):
        B,N,D=x.shape; pad=(self.bs-N%self.bs)%self.bs
        if pad: x=F.pad(x,(0,0,0,pad))
        nb=x.shape[1]//self.bs; blk=x.reshape(B,nb,self.bs,D)
        wt=F.softmax(self.w(blk).squeeze(-1),dim=-1)
        mn=torch.einsum('bnk,bnkd->bnd',wt,blk)
        sd=(torch.einsum('bnk,bnkd->bnd',wt,(blk-mn.unsqueeze(2))**2)+1e-8).sqrt()
        mx=blk.max(dim=2).values
        out=self.norm(self.proj(torch.cat([mn,sd,mx],-1)))
        return out+self.pos(torch.arange(nb,device=x.device))

class Controller(nn.Module):
    def __init__(self,dim,bs):
        super().__init__()
        self.bs,self.dim=bs,dim; self.qp=nn.Linear(dim,dim); self.kp=nn.Linear(dim,dim)
        self.temperature=GUMBEL_TEMP_START
    def forward(self,pm,state):
        s=torch.einsum('bd,bmd->bm',self.qp(state),self.kp(pm))/math.sqrt(self.dim)
        if self.training:
            sel=F.gumbel_softmax(s,tau=self.temperature,hard=True)
            idx=torch.einsum('bm,m->b',sel,torch.arange(pm.shape[1],device=s.device,dtype=torch.float))
        else: idx=s.argmax(-1).float()
        return (idx*self.bs).long(),s,idx.long()

class Foveal(nn.Module):
    def __init__(self,dim,nh,ws):
        super().__init__()
        self.ws=ws; self.attn=nn.MultiheadAttention(dim,nh,batch_first=True)
        self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def _ext(self,x,fp):
        B,N,D=x.shape; h=self.ws//2; ws=[]
        for i in range(B):
            c=fp[i].item(); s,e=max(0,c-h),min(N,c+h)
            if e-s<self.ws: s=max(0,e-self.ws) if s>0 else 0; e=min(N,s+self.ws)
            w=x[i,s:e]
            if w.shape[0]<self.ws: w=F.pad(w,(0,0,0,self.ws-w.shape[0]))
            ws.append(w)
        return torch.stack(ws)
    def forward(self,x,fp,state,acc=None):
        win=self._ext(x,fp)
        ctx=torch.cat([state.unsqueeze(1)]+([acc,win] if acc is not None else [win]),1)
        out,_=self.attn(self.n1(ctx),self.n1(ctx),self.n1(ctx)); ctx=ctx+out
        cl=ctx[:,0]; cl=cl+self.ff(self.n2(cl))
        return cl,(torch.cat([acc,win],1) if acc is not None else win)

class SaccadicBlock(nn.Module):
    def __init__(self,orig,dim,nh):
        super().__init__()
        self.ln_1=orig.ln_1; self.ln_2=orig.ln_2; self.mlp=orig.mlp
        self.pe=PeripheralEncoder(dim,BLOCK_SIZE); self.ctrl=Controller(dim,BLOCK_SIZE)
        self.fov=Foveal(dim,nh,WINDOW_SIZE)
        self.ma=nn.MultiheadAttention(dim,nh,batch_first=True); self.mn=nn.LayerNorm(dim)
        self.mg=nn.Sequential(nn.Linear(1,dim),nn.GELU(),nn.Linear(dim,1),nn.Sigmoid())
        self.op=nn.Linear(dim,dim); self.on=nn.LayerNorm(dim)
    def forward(self,x,ps=None):
        B,N,D=x.shape; res=x; h=self.ln_1(x)
        pm=self.pe(ps if ps is not None else h); st=pm.mean(1)
        fps,fls=[],[]; ac=None
        for t in range(NUM_SACCADES):
            fp,lg,_=self.ctrl(pm,st); fps.append(fp); fls.append(lg)
            st,ac=self.fov(h,fp,st,ac)
            d,_=self.ma(self.mn(pm),ac,ac)
            a=self.mg(torch.tensor([[t/NUM_SACCADES]],device=x.device,dtype=x.dtype))
            pm=pm+a*d
        x=res+self.op(self.on(st.unsqueeze(1).expand(-1,N,-1)))
        x=x+self.mlp(self.ln_2(x))
        return x,{'fixation_points':fps,'fixation_logits':fls}

class AblationModel(nn.Module):
    def __init__(self, saccadic_layers):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        dim=self.gpt2.config.n_embd; nh=self.gpt2.config.n_head
        self.max_pos=self.gpt2.config.n_positions
        self.sblocks=nn.ModuleDict()
        self.sacc_set=set(saccadic_layers)
        for i in saccadic_layers:
            self.sblocks[str(i)]=SaccadicBlock(self.gpt2.transformer.h[i],dim,nh)
        for p in self.gpt2.parameters(): p.requires_grad=False
        for b in self.sblocks.values():
            for m in [b.pe,b.ctrl,b.fov,b.ma,b.mn,b.mg,b.op,b.on]:
                for pp in m.parameters(): pp.requires_grad=True
        self.digit_heads=nn.ModuleList([nn.Linear(dim,10) for _ in range(5)])
        self.first_sacc=min(saccadic_layers) if saccadic_layers else 0
    def forward(self,ids,labels=None):
        B,N=ids.shape; dev=ids.device
        pos=torch.arange(N,device=dev).clamp(max=self.max_pos-1).unsqueeze(0)
        x=self.gpt2.transformer.drop(self.gpt2.transformer.wte(ids)+self.gpt2.transformer.wpe(pos))
        cp=torch.arange(N,device=dev); ps=None; info={}
        for i,blk in enumerate(self.gpt2.transformer.h):
            if str(i) in self.sblocks:
                x,inf=self.sblocks[str(i)](x,ps); info[i]=inf
            else:
                x=blk(x,cache_position=cp)
                if i==self.first_sacc-1: ps=x.detach()
        x=self.gpt2.transformer.ln_f(x); last=x[:,-1]
        dl=[h(last) for h in self.digit_heads]
        loss=sum(F.cross_entropy(dl[i],labels[:,i]) for i in range(5))/5 if labels is not None else None
        return {'loss':loss,'digit_logits':dl,'fixation_info':info}
    def set_gumbel_temperature(self,t):
        for b in self.sblocks.values(): b.ctrl.temperature=t
    def tp(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

def gt(step):
    p=min(step/max(GUMBEL_ANNEAL_STEPS,1),1.0)
    return GUMBEL_TEMP_START+(GUMBEL_TEMP_END-GUMBEL_TEMP_START)*p

def train_model(model, device):
    ds=PasskeyDataset(NUM_TRAIN,CONTEXT_LENGTH,seed=42)
    ld=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=AdamW(trainable,lr=LR,weight_decay=WEIGHT_DECAY)
    def lrf(s):
        if s<WARMUP_STEPS: return s/max(WARMUP_STEPS,1)
        return 0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/3000))
    sc=torch.optim.lr_scheduler.LambdaLR(opt,lrf)
    model.train(); t0=time.time(); step=0; bud=WALL_CLOCK*0.85
    while True:
        for b in ld:
            if time.time()-t0>=bud:
                log(f'  Budget: {step} steps in {time.time()-t0:.0f}s'); return
            ids=b['input_ids'].to(device); dl=b['digit_labels'].to(device)
            model.set_gumbel_temperature(gt(step))
            out=model(ids,labels=dl); loss=out['loss']
            te=torch.tensor(0.0); ec=0
            for _,inf in out['fixation_info'].items():
                for lg in inf['fixation_logits']:
                    p=F.softmax(lg,-1); te=te+(-(p*(p+1e-8).log()).sum(-1).mean()).to(te.device); ec+=1
            if ec: loss=loss-ENTROPY_BONUS*te/ec
            if SUPERVISED_WARMUP_STEPS>0 and step<SUPERVISED_WARMUP_STEPS:
                w=SUPERVISED_WARMUP_WEIGHT*(1-step/SUPERVISED_WARMUP_STEPS)
                sw=torch.tensor(0.0,device=device); sc2=0
                tgt=torch.tensor([p//BLOCK_SIZE for p in b['passkey_position']],device=device,dtype=torch.long)
                for _,inf in out['fixation_info'].items():
                    for lg in inf['fixation_logits']:
                        sw=sw+F.cross_entropy(lg,tgt.clamp(max=lg.shape[1]-1)); sc2+=1
                if sc2: loss=loss+w*sw/sc2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable,GRAD_CLIP); opt.step(); sc.step(); step+=1
            if step%50==0:
                log(f'    step {step} | loss {loss.item():.4f} | {time.time()-t0:.0f}s')

def evaluate(model, ctx_len, device):
    ds=PasskeyDataset(NUM_EVAL,ctx_len,seed=99999)
    bs=max(1,min(8,8192//ctx_len*2))
    ld=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=0,collate_fn=collate)
    model.eval(); cor=tot=0; dists=[]
    with torch.no_grad():
        for b in ld:
            ids=b['input_ids'].to(device); out=model(ids)
            for _,inf in out['fixation_info'].items():
                for fp in inf['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.append(abs(fp[i].item()-b['passkey_position'][i]))
            for i in range(ids.shape[0]):
                pred=''.join(str(d[i].argmax().item()) for d in out['digit_logits'])
                if pred==b['passkey'][i]: cor+=1
                tot+=1
    return cor/max(tot,1), sum(dists)/max(len(dists),1)

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    results=[]
    for name, sacc_layers in ABLATION_CONFIGS:
        log(f'\n{"="*60}')
        log(f'{name} — saccadic layers: {sacc_layers}')
        log(f'{"="*60}')
        model=AblationModel(sacc_layers).to(device)
        log(f'Trainable: {model.tp():,}')
        train_model(model, device)
        for ctx in [2048, 4096]:
            acc,dist=evaluate(model, ctx, device)
            log(f'  {name} @ {ctx}: accuracy={acc:.4f}, distance={dist:.2f}')
            results.append({'config':name,'saccadic_layers':sacc_layers,
                          'n_standard':12-len(sacc_layers),'n_saccadic':len(sacc_layers),
                          'eval_context':ctx,'accuracy':acc,'fixation_distance':dist})
        del model; torch.cuda.empty_cache()

    print('config\tn_standard\tn_saccadic\teval_context\taccuracy\tfixation_distance')
    for r in results:
        print(f'{r["config"]}\t{r["n_standard"]}\t{r["n_saccadic"]}\t{r["eval_context"]}\t{r["accuracy"]:.4f}\t{r["fixation_distance"]:.2f}')
    with open('layer_ablation_results.json','w') as f:
        json.dump(results,f,indent=2)

if __name__=='__main__': main()
