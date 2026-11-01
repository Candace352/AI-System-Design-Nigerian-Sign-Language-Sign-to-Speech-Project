"""YOLO26n original-distribution baseline (run ONLY after validating split manifests).
Usage: python src/train_baseline.py --data outputs/phase2/yolo_dataset/data.yaml
Requires: pip install ultralytics torch
"""
import argparse, json
from pathlib import Path
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default=None); a=p.parse_args()
    model=YOLO('yolo26n.pt')
    kwargs=dict(data=str(a.data.resolve()),epochs=a.epochs,batch=a.batch,imgsz=640,
                seed=42,deterministic=True,project='runs/nsl',name='yolo26n_original_baseline',plots=True)
    if a.device is not None:kwargs['device']=a.device
    result=model.train(**kwargs)
    root=Path(result.save_dir)
    best=root/'weights'/'best.pt'
    val=YOLO(str(best)).val(data=str(a.data.resolve()),split='val',imgsz=640,plots=True)
    summary={'best_weights':str(best),'precision':float(val.box.mp),'recall':float(val.box.mr),
             'mAP50':float(val.box.map50),'mAP50_95':float(val.box.map)}
    (root/'baseline_validation_metrics.json').write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=='__main__':main()
