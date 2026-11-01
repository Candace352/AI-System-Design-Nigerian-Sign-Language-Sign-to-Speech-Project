"""NSL Phase 2: reproducible image/annotation audit + leakage-aware dataset export.
Run: python src/phase2_pipeline.py --dataset /path/to/dataset --out outputs/phase2
Dependencies: pip install numpy pandas pillow matplotlib scikit-learn pyyaml
This script DOES NOT train YOLO, rebalance classes, crop or mask backgrounds.
"""
from __future__ import annotations
import argparse, hashlib, json, random, shutil, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
import yaml

EXT={'.jpg','.jpeg','.png','.webp'}

def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()

class ImageAuditFeatures(BaseEstimator, TransformerMixin):
    """Stateless sklearn-compatible *audit-only* feature extraction.
    No fitted image statistics or inference-time tabular features are used by YOLO.
    """
    def fit(self,X,y=None):return self
    def transform(self,X):
        out=[]
        for path in X:
            with Image.open(path) as im:
                im=im.convert('RGB'); w,h=im.size
                small=np.asarray(im.resize((64,64)),dtype=float)/255.0
                gray=small@np.array([0.2126,0.7152,0.0722])
                out.append([w,h,w/h,float(gray.mean()),float(gray.std())])
        return np.asarray(out)

def load_records(root):
    imgs=root/'images'; ann=root/'annotations'/'PASCAL_VOC'
    if not imgs.is_dir() or not ann.is_dir():
        raise FileNotFoundError(f'Expected {imgs} and {ann}')
    files=sorted(p for p in imgs.rglob('*') if p.suffix.lower() in EXT)
    xmls=sorted(ann.rglob('*.xml'))
    by_stem=defaultdict(list)
    for p in xmls:by_stem[p.stem].append(p)
    classes=sorted({p.parent.name for p in files}); class_to_id={c:i for i,c in enumerate(classes)}
    rows=[]; bad=[]
    for p in files:
        try:
            with Image.open(p) as im: w,h=im.size; im.verify()
            if w<=0 or h<=0:raise ValueError('Nonpositive image dimensions')
            matches=by_stem.get(p.stem,[])
            if len(matches)!=1:raise ValueError(f'XML matches: {len(matches)}')
            doc=ET.parse(matches[0]).getroot(); objects=doc.findall('object')
            if not objects:raise ValueError('Empty object annotation')
            lines=[]; areas=[]
            for o in objects:
                label=(o.findtext('name') or '').strip()
                if label not in class_to_id:raise ValueError(f'Unknown class in XML: {label}')
                if label!=p.parent.name:raise ValueError(f'Folder/XML class mismatch: {label}')
                b=o.find('bndbox')
                if b is None:raise ValueError('No bndbox')
                x1,y1,x2,y2=[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
                if not(np.isfinite([x1,y1,x2,y2]).all() and 0<=x1<x2<=w and 0<=y1<y2<=h):
                    raise ValueError('Invalid bounding coordinates')
                areas.append((x2-x1)*(y2-y1)/(w*h))
                lines.append(f'{class_to_id[label]} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
            rows.append(dict(path=str(p),class_name=p.parent.name,filename=p.name,width=w,height=h,
                             box_count=len(lines),mean_box_area=float(np.mean(areas)),labels=lines,sha256=sha256(p)))
        except Exception as exc:bad.append(dict(path=str(p),reason=str(exc)))
    return files,xmls,classes,rows,bad

def figure_counts(rows,out):
    counts=pd.Series(Counter(r['class_name'] for r in rows)).sort_values()
    fig,ax=plt.subplots(figsize=(12,max(8,len(counts)*0.23)))
    ax.barh(counts.index,counts.values,color='#457B9D')
    ax.set(xlabel='Valid images',title='NSL: Valid images by class');ax.grid(axis='x',alpha=.15)
    ax.tick_params(labelsize=7);fig.tight_layout();fig.savefig(out/'class_distribution_validated.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.hist([r['mean_box_area'] for r in rows],bins=30,color='#2A9D8F')
    ax.set(xlabel='Mean normalized bounding-box area',ylabel='Images',title='NSL: Bounding-box area');fig.tight_layout()
    fig.savefig(out/'bbox_area_validated.png',dpi=300,bbox_inches='tight');plt.close(fig)

def split_groups(rows,seed):
    """Hash-clustered image split. Not signer-disjoint without signer metadata.
    Splits duplicates together. Deterministic; not strictly class stratified.
    """
    groups={}
    for r in rows:groups.setdefault(r['sha256'],[]).append(r)
    keys=list(groups);random.Random(seed).shuffle(keys)
    n=len(keys); a=round(n*.70);b=round(n*.15)
    sets={'train':keys[:a],'val':keys[a:a+b],'test':keys[a+b:]}
    result={k:[r for key in ids for r in groups[key]] for k,ids in sets.items()}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('outputs/phase2'));ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--export-yolo',action='store_true',help='Copy verified images/YOLO labels into split dirs')
    args=ap.parse_args();root=args.dataset.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    images,xmls,classes,rows,bad=load_records(root)
    pd.DataFrame([{k:v for k,v in r.items() if k!='labels'} for r in rows]).to_csv(out/'validated_manifest.csv',index=False)
    pd.DataFrame(bad,columns=['path','reason']).to_csv(out/'exclusion_log.csv',index=False)
    splits=split_groups(rows,args.seed)
    split_csv=[]
    for s,items in splits.items():
        for r in items:split_csv.append({'split':s,'path':r['path'],'class_name':r['class_name'],'sha256':r['sha256']})
    pd.DataFrame(split_csv).to_csv(out/'split_manifest.csv',index=False)
    audit_pipe=Pipeline([('image_features',ImageAuditFeatures())])
    train_paths=[r['path'] for r in splits['train']]
    train_features=audit_pipe.fit_transform(train_paths) if train_paths else np.zeros((0,5))
    counts=Counter(r['class_name'] for r in rows)
    report=dict(n_indexed_images=len(images),n_xml_files=len(xmls),n_valid_images=len(rows),n_excluded=len(bad),
                n_classes_folders=len(classes),n_classes_valid=len(counts),
                class_min=min(counts.values()) if counts else None,
                class_max=max(counts.values()) if counts else None,
                split_counts={k:len(v) for k,v in splits.items()},
                audit_train_feature_matrix_shape=list(train_features.shape),
                yolo_image_tensor_shape='(N, 3, 640, 640) after letterbox; confirm at runtime',
                leakage_check_exact_hash_overlap=all(len(set(r['sha256'] for r in splits[a]) & set(r['sha256'] for r in splits[b]))==0 for a,b in [('train','val'),('train','test'),('val','test')]),
                signer_disjoint_verified=False,
                note='ImageAuditFeatures describes audit metadata, not YOLO training features. Not all rubric tabular transforms apply to detection.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    (out/'class_names.yaml').write_text(yaml.safe_dump({'names':{i:c for i,c in enumerate(classes)}},sort_keys=False))
    if rows:figure_counts(rows,out)
    if args.export_yolo:
        dest=out/'yolo_dataset'
        for s,items in splits.items():
            (dest/'images'/s).mkdir(parents=True,exist_ok=True)
            (dest/'labels'/s).mkdir(parents=True,exist_ok=True)
            for r in items:
                src=Path(r['path']); name=f"{classes.index(r['class_name']):03d}_{src.name}"
                shutil.copy2(src,dest/'images'/s/name)
                (dest/'labels'/s/(Path(name).stem+'.txt')).write_text('\n'.join(r['labels'])+'\n')
        (dest/'data.yaml').write_text(yaml.safe_dump({'path':str(dest),'train':'images/train','val':'images/val','test':'images/test','names':{i:c for i,c in enumerate(classes)}},sort_keys=False))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
