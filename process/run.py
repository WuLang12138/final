"""Run from any working directory; all artifacts stay inside this package."""
from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name,folder in [('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba')]:
    cache=ROOT/'.cache'/folder
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault(name,str(cache))
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','summary','verify','figures'])
    parser.add_argument('--dataset',choices=['DLPFC','HBC','STARmap','osmFISH'])
    parser.add_argument('--profile',choices=['paper','refined'])
    parser.add_argument('--source',choices=['current','archived'],default='current')
    parser.add_argument('--sample',nargs='+')
    parser.add_argument('--seed',nargs='+',type=int)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--inference-only',action='store_true')
    parser.add_argument('--notebook',action='store_true',help='Notebook renders the summary tables inline')
    parser.add_argument('--model-checkpoint',choices=['auto','final','best'],default='auto')
    parser.add_argument('--recover-best',action='store_true',help='Retrain missing peak weights without replacing historical final weights')
    parser.add_argument('--checkpoint',choices=['current','peak','final','prediction'],default='current')
    args=parser.parse_args()
    from src import workflow as w
    if args.action in {'verify','summary'} and args.source=='current':
        from src import current_results as current
        if args.action=='verify':return current.collect(args.dataset,args.profile,args.sample)
        return current.summarize(args.dataset,args.profile,args.sample,print_table=not args.notebook)
    if args.action=='verify':return w.verify()
    if args.action=='summary':return w.summary(print_table=not args.notebook)
    if args.action=='figures':
        print('Loading plotting libraries; the first UMAP import may take time.',flush=True)
        from src.figures import generate
        return generate(args.checkpoint,args.dataset,args.sample,args.profile)
    runs=[r for r in w.configuration()['runs'] if r['profile']==(args.profile or 'paper')
          and (not args.dataset or r['dataset']==args.dataset)
          and (not args.sample or r['sample'] in args.sample)
          and (args.seed is None or r['seed'] in args.seed)]
    if not runs:parser.error('No selected runs; refined is available only for HBC.')
    reports=[]
    for i,run in enumerate(runs,1):
        print(f'[{i}/{len(runs)}]',flush=True)
        result=w.run_one(run,args.device,args.inference_only,args.model_checkpoint,args.recover_best)
        reports.append(dict(sample=run['sample'],profile=run['profile'],seed=run['seed'],**result))
    w.write(ROOT/'results/last_execution.json',reports)
    from src.current_results import update_selection
    update_selection(reports)
    print('已更新当前实验选择；00 和 05 将读取这套方案的新预测。',flush=True)


if __name__=='__main__':
    main()
