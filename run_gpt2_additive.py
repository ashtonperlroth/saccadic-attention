"""GPT-2 Bottlenecked Additive: Frozen GPT-2 → project 768→128 → 2 saccadic layers (128-dim) → project 128→768.

Only the projection matrices and 128-dim saccadic components are trainable (~3-5M params).
This combines GPT-2's rich representations with tiny-model-speed convergence.
"""

import json, math, random, string, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

GPT2_DIM = 768; SACC_DIM = 128; N_HEADS = 4; BLOCK_SIZE = 8; WINDOW_SIZE = 64
CONTEXT_LENGTH = 2048
LR = 1e-3; WEIGHT_DECAY = 0.01; GRAD_CLIP = 1.0; BATCH_SIZE = 8
ENTROPY_BONUS = 0.01; WARMUP_STEPS = 50
SUPERVISED_WARMUP_STEPS = 300; SUPERVISED_WARMUP_WEIGHT = 2.0
GUMBEL_TEMP_START = 1.0; GUMBEL_TEMP_END = 0.1; GUMBEL_ANNEAL_STEPS = 800
NUM_TRAIN = 3000; NUM_VAL = 100; NUM_EVAL = 200
MAX_TIME = 1800; PATIENCE = 500; VAL_EVERY = 50

CONFIGS = [
    (3, 3), (3, 4),
    (4, 4), (4, 5),
    (5, 5),
]

def log(msg): print(msg, file=sys.stderr, flush=True)
_tok = GPT2Tokenizer.from_pretrained('gpt2')
ORDINALS = ['first','second','third','fourth','fifth']
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

# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiHopDataset(Dataset):
    def __init__(self, n, ctx_len, n_hops, seed=42):
        self.n_hops = n_hops; self.rng = random.Random(seed)
        self.filler_enc = [_tok.encode(" "+s) for s in FILLER]
        self.prompt_ids = _tok.encode(f" What is the {n_hops}-digit code? The code is")
        self.samples = [self._make(i, ctx_len) for i in range(n)]
    def _make(self, idx, ctx_len):
        rng = random.Random(self.rng.randint(0,2**32)+idx)
        digits = [rng.choice(string.digits) for _ in range(self.n_hops)]
        code = ''.join(digits)
        clues = [_tok.encode(f" The {ORDINALS[i]} digit of the code is {d}.") for i,d in enumerate(digits)]
        answer_ids = _tok.encode(" "+code); clue_total = sum(len(c) for c in clues)
        budget = ctx_len-clue_total-len(self.prompt_ids)-len(answer_ids)
        fill = []
        while len(fill)<budget: fill.extend(rng.choice(self.filler_enc))
        fill = fill[:budget]; seg = len(fill)//(self.n_hops+1)
        cps = []; full = list(fill)
        for i in range(self.n_hops-1,-1,-1):
            pos = min(i*seg+rng.randint(0,seg), len(full)); cps.insert(0, pos)
            full = full[:pos]+clues[i]+full[pos:]
        full = full+self.prompt_ids+answer_ids
        if len(full)>ctx_len: full=full[:ctx_len]
        elif len(full)<ctx_len: full+=[_tok.eos_token_id]*(ctx_len-len(full))
        return (torch.tensor(full[:ctx_len],dtype=torch.long),
                torch.tensor([int(d) for d in digits],dtype=torch.long),code,cps)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids,dl,code,cps = self.samples[i]
        return {'input_ids':ids,'digit_labels':dl,'passkey':code,'passkey_position':cps[0],'clue_positions':cps}

def collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'digit_labels': torch.stack([b['digit_labels'] for b in batch]),
        'passkey': [b['passkey'] for b in batch],
        'passkey_position': [b['passkey_position'] for b in batch],
        'clue_positions': [b['clue_positions'] for b in batch],
    }

# ── Saccadic Components ──────────────────────────────────────────────────────

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
        return self.norm(self.proj(torch.cat([mn,sd,mx],-1)))+self.pos(torch.arange(nb,device=x.device))

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

class SaccadicLayer(nn.Module):
    """Standalone saccadic layer operating in SACC_DIM space."""
    def __init__(self, dim, nh, n_saccades):
        super().__init__()
        self.n_saccades = n_saccades
        self.pe = PeripheralEncoder(dim, BLOCK_SIZE)
        self.ctrl = Controller(dim, BLOCK_SIZE)
        self.fov = Foveal(dim, nh, WINDOW_SIZE)
        self.ma = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.mn = nn.LayerNorm(dim)
        self.mg = nn.Sequential(nn.Linear(1,dim),nn.GELU(),nn.Linear(dim,1),nn.Sigmoid())
        self.op = nn.Linear(dim,dim); self.on = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self, x):
        B,N,D = x.shape; res = x; h = self.ln1(x)
        pm = self.pe(h); st = pm.mean(1)
        fps,fls = [],[]; ac = None
        for t in range(self.n_saccades):
            fp,lg,_ = self.ctrl(pm,st); fps.append(fp); fls.append(lg)
            st,ac = self.fov(h,fp,st,ac)
            d,_ = self.ma(self.mn(pm),ac,ac)
            a = self.mg(torch.tensor([[t/self.n_saccades]],device=x.device,dtype=x.dtype))
            pm = pm+a*d
        x = res+self.op(self.on(st.unsqueeze(1).expand(-1,N,-1)))
        x = x+self.mlp(self.ln2(x))
        return x, {'fixation_points':fps,'fixation_logits':fls}



# ── Additive Model ───────────────────────────────────────────────────────────

class GPT2Additive(nn.Module):
    """Frozen GPT-2 (12 layers) → project 768→128 → 2 saccadic layers (128-dim) → project 128→768."""
    def __init__(self, n_hops, n_saccades):
        super().__init__()
        self.n_hops = n_hops
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.max_pos = self.gpt2.config.n_positions

        # Freeze ALL GPT-2 params
        for p in self.gpt2.parameters(): p.requires_grad = False

        # Bottleneck projections (trainable)
        self.proj_down = nn.Linear(GPT2_DIM, SACC_DIM)  # 768 → 128
        self.proj_up = nn.Linear(SACC_DIM, GPT2_DIM)    # 128 → 768

        # 2 saccadic layers in 128-dim space (trainable)
        self.sacc1 = SaccadicLayer(SACC_DIM, N_HEADS, n_saccades)
        self.sacc2 = SaccadicLayer(SACC_DIM, N_HEADS, n_saccades)

        # Classification head from 768-dim (after proj_up + residual)
        self.ln_out = nn.LayerNorm(GPT2_DIM)
        self.digit_heads = nn.ModuleList([nn.Linear(GPT2_DIM, 10) for _ in range(n_hops)])

        self._tp = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, ids, labels=None):
        B, N = ids.shape; dev = ids.device
        # Run frozen GPT-2
        pos = torch.arange(N, device=dev).clamp(max=self.max_pos-1).unsqueeze(0)
        x = self.gpt2.transformer.drop(self.gpt2.transformer.wte(ids) + self.gpt2.transformer.wpe(pos))
        cp = torch.arange(N, device=dev)
        for blk in self.gpt2.transformer.h:
            x = blk(x, cache_position=cp)
        x_gpt2 = self.gpt2.transformer.ln_f(x)  # (B, N, 768)

        # Project down to 128-dim
        x_small = self.proj_down(x_gpt2)  # (B, N, 128)

        # 2 saccadic layers in compressed space
        x_small, info1 = self.sacc1(x_small)
        x_small, info2 = self.sacc2(x_small)

        # Project back up and residual with GPT-2 output
        x_out = x_gpt2 + self.proj_up(x_small)  # (B, N, 768)

        # Classify
        last = self.ln_out(x_out[:, -1])
        dl = [h(last) for h in self.digit_heads]
        loss = sum(F.cross_entropy(dl[i], labels[:,i]) for i in range(self.n_hops))/self.n_hops if labels is not None else None
        return {'loss': loss, 'digit_logits': dl, 'fixation_info': {0: info1, 1: info2}}

    def set_gumbel_temperature(self, t):
        self.sacc1.ctrl.temperature = t
        self.sacc2.ctrl.temperature = t


# ── Training ──────────────────────────────────────────────────────────────────

def gt(step):
    p = min(step/max(GUMBEL_ANNEAL_STEPS,1),1.0)
    return GUMBEL_TEMP_START+(GUMBEL_TEMP_END-GUMBEL_TEMP_START)*p

def quick_val(model, vl, n_hops, device):
    model.eval(); cor=tot=0
    with torch.no_grad():
        for b in vl:
            ids=b['input_ids'].to(device); out=model(ids)
            for i in range(ids.shape[0]):
                pred=''.join(str(d[i].argmax().item()) for d in out['digit_logits'])
                if pred==b['passkey'][i]: cor+=1
                tot+=1
    model.train(); return cor/max(tot,1)

def train_converge(model, n_hops, device):
    tds = MultiHopDataset(NUM_TRAIN, CONTEXT_LENGTH, n_hops, seed=42)
    vds = MultiHopDataset(NUM_VAL, CONTEXT_LENGTH, n_hops, seed=77777)
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(vds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    def lrf(s):
        if s<WARMUP_STEPS: return s/max(WARMUP_STEPS,1)
        return 0.5*(1+math.cos(math.pi*(s-WARMUP_STEPS)/5000))
    sc = torch.optim.lr_scheduler.LambdaLR(opt, lrf)
    model.train(); t0=time.time(); step=0; best=0.0; since=0; reason='max_time'
    while True:
        for b in tl:
            elapsed=time.time()-t0
            if elapsed>=MAX_TIME:
                reason=f'max_time ({MAX_TIME}s)'; log(f'  MAX TIME: {step} steps, best={best:.4f}')
                return step,elapsed,best,reason
            if step>0 and step%VAL_EVERY==0:
                va=quick_val(model,vl,n_hops,device)
                if va>best:
                    best=va; since=0
                    log(f'  step {step} | val={va:.4f} (NEW BEST) | {elapsed:.0f}s')
                    if va>=1.0:
                        reason='perfect_accuracy'; log(f'  PERFECT at step {step}')
                        return step,elapsed,best,reason
                else:
                    since+=VAL_EVERY
                    if step%200==0: log(f'  step {step} | val={va:.4f} (best={best:.4f}, plat={since}) | {elapsed:.0f}s')
                if since>=PATIENCE:
                    reason=f'plateau ({PATIENCE} steps)'; log(f'  PLATEAU at step {step}, best={best:.4f}')
                    return step,elapsed,best,reason
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
                for _,inf in out['fixation_info'].items():
                    for si,lg in enumerate(inf['fixation_logits']):
                        ci=si%n_hops
                        tgt=torch.tensor([cp[ci]//BLOCK_SIZE for cp in b['clue_positions']],
                                        device=device,dtype=torch.long).clamp(max=lg.shape[1]-1)
                        sw=sw+F.cross_entropy(lg,tgt); sc2+=1
                if sc2: loss=loss+w*sw/sc2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable,GRAD_CLIP); opt.step(); sc.step(); step+=1

def final_eval(model, n_hops, device):
    ds=MultiHopDataset(NUM_EVAL,CONTEXT_LENGTH,n_hops,seed=99999)
    ld=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate)
    model.eval(); cor=tot=0; dists=[]
    with torch.no_grad():
        for b in ld:
            ids=b['input_ids'].to(device); out=model(ids)
            for _,inf in out['fixation_info'].items():
                for fp in inf['fixation_points']:
                    for i in range(ids.shape[0]):
                        dists.append(min(abs(fp[i].item()-cp) for cp in b['clue_positions'][i]))
            for i in range(ids.shape[0]):
                pred=''.join(str(d[i].argmax().item()) for d in out['digit_logits'])
                if pred==b['passkey'][i]: cor+=1
                tot+=1
    return cor/max(tot,1), sum(dists)/max(len(dists),1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    results=[]
    for n_hops,n_saccades in CONFIGS:
        log(f'\n{"="*60}')
        log(f'ADDITIVE: N_HOPS={n_hops}, N_SACCADES={n_saccades}')
        log(f'{"="*60}')
        model=GPT2Additive(n_hops,n_saccades).to(device)
        log(f'Trainable: {model._tp:,} (GPT-2 frozen, 2 saccadic layers trainable)')
        steps,elapsed,best_val,reason=train_converge(model,n_hops,device)
        acc,dist=final_eval(model,n_hops,device)
        log(f'FINAL: acc={acc:.4f}, dist={dist:.2f}, steps={steps}, time={elapsed:.0f}s, reason={reason}')
        results.append({'n_hops':n_hops,'n_saccades':n_saccades,'accuracy':acc,
                       'avg_clue_distance':dist,'steps':steps,'time_s':round(elapsed,1),
                       'converge_reason':reason,'architecture':'additive'})
        del model; torch.cuda.empty_cache()

    print('n_hops\tn_saccades\taccuracy\tavg_clue_distance\tsteps\ttime_s\tconverge_reason')
    for r in results:
        print(f'{r["n_hops"]}\t{r["n_saccades"]}\t{r["accuracy"]:.4f}\t{r["avg_clue_distance"]:.2f}\t{r["steps"]}\t{r["time_s"]}\t{r["converge_reason"]}')
    with open('gpt2_additive_results.json','w') as f:
        json.dump(results,f,indent=2)

if __name__=='__main__': main()
