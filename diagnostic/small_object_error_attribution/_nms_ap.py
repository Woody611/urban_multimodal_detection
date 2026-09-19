"""只读：对 baseline 与 NMS 变体统一计算官方 mAP + Small/Medium/Large AP50-95。"""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import official_map as OM
SPLIT=ROOT/"data/processed/rgbid_split"; THR=[float(t) for t in OM.IOUV_OFFICIAL]; SM,MM=1024,9216
def bk(b):
    a=np.clip(b[:,2]-b[:,0],0,None)*np.clip(b[:,3]-b[:,1],0,None)
    r=np.full(len(a),2,int); r[a<SM]=0; r[(a>=SM)&(a<MM)]=1; return r
pairs=[a.split("=",1) for a in sys.argv[1:]]
gt,_,_=OM.collect(SPLIT/"images/val/visible",SPLIT/"labels/val/visible",ROOT/pairs[0][1]/"results",0.0,True)
res={}
for nm,d in pairs:
    p=ROOT/d
    if not (p/"results").is_dir(): continue
    g,preds,st=OM.collect(SPLIT/"images/val/visible",SPLIT/"labels/val/visible",p/"results",0.0,True)
    off=OM.evaluate(g,preds,"mean","zero","conf")
    sz={}
    for si,sn in enumerate(("small","medium","large")):
        aps=[]
        for t in THR:
            v=[]
            for c in range(12):
                if len(g[c])==0: continue
                sub={}
                for img,bx in g[c].items():
                    m=bk(bx)==si
                    if m.any(): sub[img]=bx[m]
                if not sub: continue
                r2,p2=OM.curve_for_class(preds[c],sub,t,match="conf")
                v.append(OM.ap_from_curve(r2,p2,avg="mean",tail="zero"))
            aps.append(float(np.mean(v)) if v else 0.0)
        sz[sn]=float(np.mean(aps))
    res[nm]=dict(m50=off["mAP50"],m75=off["mAP75"],m=off["mAP50-95"],sz=sz,pred=st["pred"])
print("="*100); print("§9 NMS / max_det 敏感性（官方口径；只改推理 NMS 参数，模型与验证集不变）"); print("="*100)
print(f"  {'配置':<22}{'mAP50':>9}{'mAP75':>9}{'mAP50-95':>10}{'Small':>10}{'Medium':>10}{'Large':>10}{'框数':>7}")
for k,v in res.items():
    print(f"  {k:<22}{v['m50']:>9.5f}{v['m75']:>9.5f}{v['m']:>10.5f}{v['sz']['small']:>10.5f}{v['sz']['medium']:>10.5f}{v['sz']['large']:>10.5f}{v['pred']:>7}")
base=pairs[0][0]
print(f"\n  相对 {base}：")
print(f"  {'配置':<22}{'ΔmAP50-95':>12}{'ΔSmall':>10}{'ΔMedium':>10}{'ΔLarge':>10}")
b=res[base]
for k,v in res.items():
    if k==base: continue
    print(f"  {k:<22}{v['m']-b['m']:>+12.5f}{v['sz']['small']-b['sz']['small']:>+10.5f}"
          f"{v['sz']['medium']-b['sz']['medium']:>+10.5f}{v['sz']['large']-b['sz']['large']:>+10.5f}")
print("="*100)
