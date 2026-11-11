"""Organize existing saved artifacts without changing experiment contents."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import shutil
import sys

BASE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan = []
    new_runs = sorted(p for p in BASE.glob('tuning_*') if p.is_dir())
    batch_name = 'configured_batch_' + new_runs[0].name.removeprefix('tuning_') if new_runs else 'none'
    for src in sorted(BASE.iterdir()):
        if not src.is_dir() or not (src.name == '1st' or re.fullmatch(r'(tuning_)?\d{4}-\d{2}-\d{2}_[\d-]+', src.name)):
            continue
        summaries=list(src.glob('*summary.txt'))
        text=summaries[0].read_text(encoding='utf-8') if summaries else ''
        cfg={}
        start=text.find("{'paths':")
        if start >= 0:
            cfg=ast.literal_eval(text[start:text.rfind('}')+1])
        validation=cfg.get('validation',{})
        strategy=validation.get('strategy')
        if not strategy:
            match=re.search(r'^Validation: (\w+)',text,re.M)
            strategy=match.group(1) if match else 'unknown'
        imputation=cfg.get('preprocessing',{}).get('numeric_imputation',{}).get('strategy','unknown')
        model=cfg.get('hyperparameter_tuning',{}).get('model','unknown')
        years=validation.get('sliding_window',{}).get('train_years','unknown')
        if src.name.startswith('tuning_'):
            complete=all((src/n).is_file() for n in ['tuning_summary.txt','comparison_summary.csv','outer_results.csv','trials.csv'])
            batch=src.name >= 'tuning_2026-11-11_21-33' and src.name <= 'tuning_2026-11-11_22-13'
            status='complete' if complete else ('empty' if not list(src.iterdir()) else 'incomplete')
            if batch:
                group=Path('tuning/configured_sliding_batch_2026-11-11')/f'{years}_years'/model
            elif src.name < 'tuning_2026-11-11_21-33':
                group=Path('tuning/earlier_attempts')/status
            else:
                group=Path('tuning')/batch_name/strategy
                if strategy == 'sliding_window':
                    group=group/f'{years}_years'
                group=group/model
                if not complete:
                    group=group/status
        else:
            status='saved_evaluation'
            group=Path('baselines')/strategy
            if strategy=='sliding_window':
                group=group/f'{years}_years'
            group=group/f'imputation_{imputation}'
        dst=BASE/group/src.name
        assert src.resolve().is_relative_to(BASE.resolve())
        assert dst.resolve().is_relative_to(BASE.resolve()) and not dst.exists()
        hashes={str(p.relative_to(src)):digest(p) for p in src.rglob('*') if p.is_file()}
        plan.append((src,dst,hashes,dict(original_path=src.name,new_path=dst.relative_to(BASE).as_posix(),status=status,configured_strategy=strategy,configured_window_years=years if strategy=='sliding_window' else '',imputation=imputation,model=model if src.name.startswith('tuning_') else 'multiple',file_count=len(hashes))))
    if not plan:
        print('No new experiment directories to organize.')
        return
    for src,dst,_,row in plan:
        print(f'{src.name} -> {row["new_path"]} ({row["status"]}, {row["file_count"]} files)')
    if '--dry-run' in sys.argv:
        return
    index=BASE/'experiment_index.csv'
    previous=list(csv.DictReader(index.open(encoding='utf-8',newline=''))) if index.exists() else []
    manifest_path=BASE/'organization_manifest.json'
    previous_manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    assert not ({r['new_path'] for r in previous} & {r['new_path'] for _,_,_,r in plan})
    for src,dst,hashes,row in plan:
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(src),str(dst))
        assert hashes=={str(p.relative_to(dst)):digest(p) for p in dst.rglob('*') if p.is_file()}
    with (BASE/'experiment_index.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(plan[0][3]))
        writer.writeheader(); writer.writerows(previous); writer.writerows(row for _,_,_,row in plan)
    manifest_path.write_text(json.dumps(previous_manifest+[dict(**row,sha256=hashes) for _,_,hashes,row in plan],indent=2),encoding='utf-8')
    # Repair provenance paths in the existing analysis without recalculating metrics.
    manifest=BASE/'sliding_tuning_analysis/source_manifest.json'
    if manifest.exists():
        entries=json.loads(manifest.read_text(encoding='utf-8'))
        for entry in entries:
            for src,dst,_,_ in plan:
                old='results/'+src.name+'/'
                if entry['path'].replace('\\','/').startswith(old):
                    entry['path']='results/'+dst.relative_to(BASE).as_posix()+'/'+Path(entry['path']).name
                    assert digest(BASE.parent/entry['path'])==entry['sha256']
                    break
        manifest.write_text(json.dumps(entries,indent=2),encoding='utf-8')
    print(f'Moved and hash-verified {len(plan)} directories, {sum(len(h) for _,_,h,_ in plan)} files.')

if __name__=='__main__':
    main()
