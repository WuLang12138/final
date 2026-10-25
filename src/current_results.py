"""Validated current predictions, independent of historical result archives."""
import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from . import workflow as w


def update_selection(reports):
    path=w.ROOT/'results/current_selection.json'
    selection=w.read(path) if path.exists() else {'schema':1,'datasets':{}}
    config=w.configuration()
    by_identity={(r['sample'],r['profile'],r['seed']):r for r in config['runs']}
    grouped={}
    for report in reports:
        run=by_identity[(report['sample'],report['profile'],report['seed'])]
        grouped.setdefault(run['dataset'],[]).append(report)
    for dataset,group in grouped.items():
        profiles={r['profile'] for r in group};kinds={r['checkpoint'] for r in group}
        if len(profiles)!=1:raise ValueError('一次当前选择不能混用同一数据集的多个方案')
        selection['datasets'][dataset]=dict(profile=next(iter(profiles)),
            checkpoint=next(iter(kinds)) if len(kinds)==1 else 'mixed',source='last_successful_run',
            note='选择最近成功运行的方案；汇总时仍要求该方案全部样本及 seed 完整。')
    w.write(path,selection)


def selected_runs(dataset=None,profile=None,samples=None):
    config=w.configuration()
    path=w.ROOT/'results/current_selection.json'
    if not path.exists():
        raise FileNotFoundError('尚无当前实验选择记录。请先运行相应数据 Notebook；不会回退到历史归档。')
    selection=w.read(path)['datasets'];runs=[]
    for ds in config['datasets']:
        if dataset and ds!=dataset:continue
        if ds not in selection:raise ValueError('缺少当前实验选择：'+ds)
        chosen=profile if profile is not None else selection[ds]['profile']
        kind=selection[ds]['checkpoint'] if chosen==selection[ds]['profile'] else 'best'
        if kind not in {'best','final'}:
            raise ValueError(f'{ds} 最近预测混用了峰值和末轮模型，请统一选择后运行；不会混合汇总。')
        expected=[r for r in config['runs'] if r['dataset']==ds and r['profile']==chosen
                  and (not samples or r['sample'] in samples)]
        if not expected:raise ValueError(f'{ds} 没有方案 {chosen}')
        for r in expected:r=dict(r);r['selected_checkpoint']=kind;runs.append(r)
    if not runs:raise ValueError('没有匹配的实验')
    identities=[(r['dataset'],r['sample'],r['profile'],r['seed']) for r in runs]
    if len(set(identities))!=len(identities):raise ValueError('当前实验配置存在重复项')
    for sample in {r['sample'] for r in runs}:
        group=[r for r in runs if r['sample']==sample]
        if sorted(r['seed'] for r in group)!=sorted(config['seeds']):
            raise ValueError('当前样本 seed 不完整：'+sample)
    return runs


def collect(dataset=None,profile=None,samples=None):
    rows=[];errors=[];input_cache={}
    runs=selected_runs(dataset,profile,samples)
    for i,run in enumerate(runs,1):
        try:
            kind=run['selected_checkpoint'];best=kind=='best'
            weight=w.best_weight_path(run) if best else w.local(run['weight'])
            dest=w.local(run['prediction'])/('best' if best else '')
            report_file=dest/'metrics.json';prediction_file=dest/'prediction.npz'
            for path in [weight,report_file,prediction_file]:
                if not path.is_file():raise FileNotFoundError(str(path.relative_to(w.ROOT)))
            report=w.read(report_file)
            if report['checkpoint']!=kind or w.local(report['weight'])!=weight:
                raise ValueError('预测报告引用了其他类型或路径的模型')
            digest=w.sha(weight)
            if report['weight_sha256']!=digest:raise ValueError('权重已改变，预测报告过期，请重新预测')
            state=torch.load(weight,map_location='cpu',weights_only=True)
            w.validate_weight(run,state,kind)
            if report['epoch']!=state['epoch']:raise ValueError('预测轮次与权重轮次不一致')
            inp=w.local(run['input']);cache_key=(run['input'],tuple(sorted(run['input_sha256'].items())))
            if cache_key not in input_cache:
                for name,h in run['input_sha256'].items():
                    if w.sha(inp/name)!=h:raise ValueError('输入文件校验失败：'+name)
                a=ad.read_h5ad(inp/'expression.h5ad',backed='r')
                try:
                    input_cache[cache_key]=(a.obs_names.to_numpy(dtype=str),
                        a.obs['reference'].astype(str).to_numpy(dtype=str),np.asarray(a.obsm['spatial']).copy())
                finally:a.file.close()
            ids,reference,spatial=input_cache[cache_key]
            with np.load(prediction_file,allow_pickle=False) as z:
                if not (np.array_equal(z['obs_names'],ids) and np.array_equal(z['reference'],reference)
                        and np.array_equal(z['spatial'],spatial)):
                    raise ValueError('预测的检测点顺序、参考标签或坐标与输入不匹配')
                if z['embedding'].shape!=(len(ids),run['parameters']['latent_dim']) or not np.isfinite(z['embedding']).all():
                    raise ValueError('预测表示尺寸或数值异常')
                if z['prediction'].shape!=(len(ids),) or int(z['epoch'])!=state['epoch']:
                    raise ValueError('预测标签尺寸或轮次异常')
                ari=float(adjusted_rand_score(reference,z['prediction']))
                nmi=float(normalized_mutual_info_score(reference,z['prediction']))
            if not np.allclose([ari,nmi],[report['metrics']['ARI'],report['metrics']['NMI']],atol=1e-10,rtol=0):
                raise ValueError('预测标签重算的指标与报告不一致')
            history_file=None;history_peak=None;history_final=None
            if best:
                peak_result=w.local(state['peak_result']);history_file=peak_result.parent/'history.csv'
                if not peak_result.exists() or w.sha(peak_result)!=state['peak_result_sha256']:
                    raise ValueError('新训练峰值记录缺失或被修改')
                h=pd.read_csv(history_file)
                if len(h)!=run['parameters']['epochs'] or list(h.epoch)!=list(range(1,len(h)+1)) or not np.isfinite(h.eval_ARI).all():
                    raise ValueError('新训练历史未完整覆盖指定训练预算')
                history_peak=float(h.eval_ARI.max());history_final=float(h.eval_ARI.iloc[-1])
                if abs(history_peak-state['metrics']['ARI'])>1e-10 or int(h.loc[h.eval_ARI.idxmax(),'epoch'])!=state['epoch']:
                    raise ValueError('峰值权重不是新训练记录中首次达到最高 ARI 的轮次')
            rows.append(dict(dataset=run['dataset'],sample=run['sample'],donor=run.get('donor'),
                profile=run['profile'],seed=run['seed'],checkpoint=kind,epoch=state['epoch'],ARI=ari,NMI=nmi,
                training_peak_ARI=history_peak,training_final_ARI=history_final,
                reload_ARI_delta=ari-history_peak if best else None,
                weight=weight.relative_to(w.ROOT).as_posix(),weight_sha256=digest,
                prediction=prediction_file.relative_to(w.ROOT).as_posix(),prediction_sha256=w.sha(prediction_file),
                report=report_file.relative_to(w.ROOT).as_posix(),report_sha256=w.sha(report_file),
                history=history_file.relative_to(w.ROOT).as_posix() if history_file else None,
                history_sha256=w.sha(history_file) if history_file else None))
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            errors.append(dict(dataset=run['dataset'],sample=run['sample'],profile=run['profile'],seed=run['seed'],error=str(exc)))
        if i%15==0 or i==len(runs):print(f'核验当前预测 {i}/{len(runs)}',flush=True)
    resolved_selection={'datasets':{r['dataset']:dict(profile=r['profile'],checkpoint=r['selected_checkpoint']) for r in runs}}
    manifest=dict(source='current_predictions',archive_fallback=False,expected_runs=len(runs),
        validated_runs=len(rows),passed=not errors,errors=errors,selection=resolved_selection,runs=rows)
    w.write(w.ROOT/'results/current_verification.json',manifest)
    if errors:
        lines=[f"{r['dataset']}/{r['sample']}/{r['profile']}/seed{r['seed']}: {r['error']}" for r in errors]
        raise RuntimeError('当前实验结果不完整或不一致；没有使用历史结果填补。\n'+'\n'.join(lines))
    return pd.DataFrame(rows),manifest


def summarize(dataset=None,profile=None,samples=None,print_table=True):
    df,manifest=collect(dataset,profile,samples)
    keys=['dataset','sample','profile','checkpoint']
    per_sample=df.groupby(keys,sort=False).agg(seeds=('seed','count'),mean_ARI=('ARI','mean'),sd_ARI=('ARI','std'),
        mean_NMI=('NMI','mean'),sd_NMI=('NMI','std')).reset_index()
    aggregate=[]
    for (ds,pr,ck),g in df.groupby(['dataset','profile','checkpoint'],sort=False):
        v=g.groupby('sample').ARI.mean() if ds=='DLPFC' else g.ARI
        aggregate.append(dict(dataset=ds,profile=pr,checkpoint=ck,n=len(v),
            unit='slice_seed_mean' if ds=='DLPFC' else 'seed',mean_ARI=float(v.mean()),
            median_ARI=float(v.median()),sd_ARI=float(v.std(ddof=1))))
    out=w.ROOT/'results/tables/current';out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'seed_metrics.csv',index=False,encoding='utf-8-sig')
    per_sample.to_csv(out/'sample_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(aggregate).to_csv(out/'dataset_metrics.csv',index=False,encoding='utf-8-sig')
    w.write(out/'source_manifest.json',manifest)
    if print_table:print(per_sample.to_string(index=False),flush=True)
    return df
