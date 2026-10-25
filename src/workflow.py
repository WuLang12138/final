"""Main-method inference and fixed-budget training, with portable local paths."""
from pathlib import Path
import hashlib
import json
import random

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def local(path):
    p = (ROOT/path).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError('Project paths must remain inside this repository: '+str(path))
    return p


def configuration():
    return read(ROOT/'config/paper.json')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sparse_tensor(matrix, device):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(np.vstack((coo.row,coo.col)), coo.data,
        coo.shape, dtype=torch.float32, device=device).coalesce()


def inputs(run):
    directory = local(run['input'])
    for name, digest in run['input_sha256'].items():
        if sha(directory/name) != digest:
            raise ValueError(f'Prepared input changed: {directory/name}; weights cannot be reused silently.')
    a = ad.read_h5ad(directory/'expression.h5ad')
    A = sp.load_npz(directory/'adjacency.npz').tocsr()
    P = sp.load_npz(directory/'transition.npz').tocsr()
    if A.shape != (len(a),len(a)) or P.shape != A.shape or (A != A.T).nnz or A.diagonal().any():
        raise ValueError('Invalid spatial graph')
    expected = A + sp.eye(len(a))
    expected = sp.diags(1/np.asarray(expected.sum(1)).ravel()) @ expected
    delta = (P-expected).tocoo()
    if delta.nnz and abs(delta.data).max() > 1e-6:
        raise ValueError('Transition must equal row_normalize(A + I)')
    from sklearn.neighbors import kneighbors_graph
    knn=kneighbors_graph(np.asarray(a.obsm['spatial']),run['parameters']['spatial_k'],
                         mode='connectivity',include_self=False)
    knn=knn.maximum(knn.T).tocsr();knn.setdiag(0);knn.eliminate_zeros()
    if (knn != A).nnz:
        raise ValueError('Stored adjacency is not the configured KNN8 union graph')
    return a, A, P


def build(params, n_features):
    from model.propagation import SiDMGF
    args = dict(nfeat=n_features, nhid1=params['hidden_dim'], nhid2=params['latent_dim'],
                dropout=params['dropout'], signal_variant='spatial_only')
    if params['encoder']=='appnp':
        return SiDMGF(**args, encoder_type='appnp', propagation_steps=params['steps'],
                      propagation_alpha=params['alpha'])
    if params['encoder']=='acm_split':
        from model.acm import AdaptedSiDMGF
        return AdaptedSiDMGF(**args, encoder_type='acm_style', framework='acm_split_noaux')
    raise ValueError('Only the paper APPNP and ACM split encoders are exposed')


def embedding(model, x, p):
    return model.MLP(model.dropout(model.GNN_S(x,p)))


def evaluate(model, x, p, labels, n_clusters, seed):
    model.eval()
    with torch.no_grad():
        z = embedding(model,x,p).cpu().numpy()
    if not np.isfinite(z).all(): raise RuntimeError('Non-finite model embedding')
    with threadpool_limits(limits=1):
        prediction = KMeans(n_clusters=n_clusters,n_init=20,random_state=seed).fit_predict(z)
    return z, prediction, dict(ARI=float(adjusted_rand_score(labels,prediction)),
        NMI=float(normalized_mutual_info_score(labels,prediction)))


def save_arrays(path, a, z, prediction, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temp, embedding=z, prediction=np.asarray(prediction),
        reference=a.obs['reference'].astype(str).to_numpy(dtype=str),
        obs_names=a.obs_names.to_numpy(dtype=str), spatial=np.asarray(a.obsm['spatial']),
        epoch=np.int32(epoch))
    temp.replace(path)


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(value,temp)
    temp.replace(path)


def best_weight_path(run):
    path=local(run['weight'])
    return path.with_name(path.stem+'_best.pt')


def weight_metadata(run, model, epoch, kind='final'):
    return dict(format=1, model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        parameters=run['parameters'],sample=run['sample'],seed=run['seed'],epoch=epoch,
        checkpoint_kind=kind,input_sha256=run['input_sha256'])


def validate_weight(run, ckpt, kind='final', require_complete=True):
    for key in ['parameters','seed','sample','input_sha256']:
        if ckpt.get(key) != run[key]:
            raise ValueError(f'Existing checkpoint {key} does not match configuration. '
                             'Use a separate configuration/output for new experiments.')
    if ckpt.get('format') != 1 or ckpt.get('checkpoint_kind') != kind:
        raise ValueError('Unsupported checkpoint format/kind')
    if kind=='final' and ckpt.get('epoch') != run['parameters']['epochs']:
        raise ValueError('Incomplete checkpoint is not a final model')
    if kind=='best':
        if not 1 <= ckpt.get('epoch',0) <= run['parameters']['epochs']:
            raise ValueError('Invalid peak epoch')
        if not np.isfinite(ckpt.get('metrics',{}).get('ARI',np.nan)):
            raise ValueError('Peak checkpoint has no valid ARI')
        if require_complete and (not ckpt.get('training_complete') or
                                  ckpt.get('completed_epochs')!=run['parameters']['epochs']):
            raise ValueError('峰值权重来自尚未完成的训练，请继续相应训练，不能当作完整实验。')


def completed_best(run):
    path=best_weight_path(run)
    if not path.exists(): return False
    state=torch.load(path,map_location='cpu',weights_only=True)
    validate_weight(run,state,'best',require_complete=False)
    return bool(state.get('training_complete') and state.get('completed_epochs')==run['parameters']['epochs'])


def structure_loss(pred, target):
    mask = ~torch.eye(len(target),dtype=torch.bool,device=target.device)
    positives = target[mask].sum()
    negatives = mask.sum()-positives
    weight = (negatives/(positives+1e-8)).clamp(max=100)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight,reduction='none')(pred,target.float())
    return loss[mask].mean()


def train_missing(run, model, a, A, x, p, device, recover_best=False):
    """Resume interrupted local training; never overwrite archived paper results."""
    from model.propagation import ZINB
    params=run['parameters']; seed=run['seed']; labels=a.obs['reference'].astype(str).to_numpy()
    prefix='peak_recovery_seed_' if recover_best else 'training_seed_'
    dest=local(run['prediction']).parent/(prefix+str(seed))
    dest.mkdir(parents=True,exist_ok=True)
    resume=dest/'resume.pt'; rows=[]; start=1; best=-float('inf');best_state=None;best_arrays=None
    best_path=best_weight_path(run)
    final_path=local(run['weight'])
    if recover_best: final_path=final_path.with_name(final_path.stem+'_retrained_final.pt')
    optimizer=torch.optim.Adam(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
    if resume.exists():
        # This file is created only by this local training function; it includes RNG state.
        state=torch.load(resume,map_location=device,weights_only=False)
        if state.get('peak_checkpoint_schema')!=1:
            raise ValueError('旧训练断点未保存峰值权重，无法恢复已经过去的峰值。请启用补存峰值的独立重跑。')
        if state['parameters']!=params or state['input_sha256']!=run['input_sha256'] or state['seed']!=seed:
            raise ValueError('Interrupted training belongs to a different configuration')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        rows=state['history'];start=state['epoch']+1;best=state['best']
        best_state=state['best_checkpoint'];best_arrays=state['best_arrays']
        validate_weight(run,best_state,'best',require_complete=False)
        # Roll back an uncommitted peak written after the last resumable epoch.
        save_torch(best_path,best_state)
        save_arrays(dest/'peak.npz',a,best_arrays['z'],best_arrays['prediction'],best_state['epoch'])
        random.setstate(state['python']);np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if device.startswith('cuda') and state['cuda'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
        print(f'Resume at epoch {start}',flush=True)
    target=torch.as_tensor(A.toarray(),device=device,dtype=torch.float32)
    for epoch in range(start,params['epochs']+1):
        model.train();optimizer.zero_grad()
        _,rec,pi,disp,mean,_=model(x,p,p)
        loss=ZINB(pi,theta=disp,ridge_lambda=0).loss(x,mean,mean=True)+params['beta']*structure_loss(rec,target)
        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch {epoch}')
        loss.backward();optimizer.step()
        z,pred,metrics=evaluate(model,x,p,labels,params['clusters'],seed)
        rows.append(dict(epoch=epoch,loss=float(loss.detach()),eval_ARI=metrics['ARI'],eval_NMI=metrics['NMI']))
        if metrics['ARI']>best:
            best=metrics['ARI']
            best_state=weight_metadata(run,model,epoch,kind='best')
            best_state.update(metrics=metrics,training_complete=False,completed_epochs=epoch,
                selection='maximum post-update ARI; reference-label-assisted; earliest epoch on ties',
                peak_result=(dest/'peak.npz').relative_to(ROOT).as_posix())
            best_arrays=dict(z=z.copy(),prediction=pred.copy())
            # CPU tensor clones preserve the exact peak even as optimization continues.
            save_torch(best_path,best_state)
            save_arrays(dest/'peak.npz',a,z,pred,epoch)
            print(f'保存峰值模型：epoch={epoch}, ARI={best:.6f}, {best_path.name}',flush=True)
        if epoch%10==0 or epoch==params['epochs']:
            save_torch(resume,dict(parameters=params,input_sha256=run['input_sha256'],seed=seed,
                model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=rows,best=best,
                peak_checkpoint_schema=1,best_checkpoint=best_state,best_arrays=best_arrays,
                python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
            pd.DataFrame(rows).to_csv(dest/'history.csv',index=False)
        if epoch%25==0 or epoch==params['epochs']:
            print(f'epoch {epoch}/{params["epochs"]}: ARI={metrics["ARI"]:.6f}',flush=True)
    if best_state is None: raise RuntimeError('No peak checkpoint was produced')
    state=weight_metadata(run,model,params['epochs'])
    save_torch(final_path,state)
    save_arrays(dest/'peak.npz',a,best_arrays['z'],best_arrays['prediction'],best_state['epoch'])
    best_state.update(training_complete=True,completed_epochs=params['epochs'],
                      peak_result_sha256=sha(dest/'peak.npz'))
    save_torch(best_path,best_state)
    write(dest/'completed.json',dict(peak_epoch=best_state['epoch'],peak=best_state['metrics'],
        final_epoch=params['epochs'],final_weight=final_path.relative_to(ROOT).as_posix(),
        best_weight=best_path.relative_to(ROOT).as_posix(),recover_best=recover_best))
    if resume.exists(): resume.unlink()  # Final weight and complete history are safely written.
    return state


def run_one(run, device='auto', inference_only=False, model_checkpoint='auto', recover_best=False):
    if model_checkpoint not in {'auto','final','best'}: raise ValueError('Unknown model checkpoint')
    if recover_best and (inference_only or model_checkpoint=='final'):
        raise ValueError('补存峰值训练不能同时使用仅预测或指定末轮模型。')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(run['parameters']['threads']);seed_all(run['seed'])
    a,A,P=inputs(run);X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    x=torch.as_tensor(X,dtype=torch.float32,device=device);p=sparse_tensor(P,device)
    model=build(run['parameters'],a.n_vars).to(device)
    final_path=local(run['weight']);has_best=completed_best(run);action='loaded_existing_weight'
    if recover_best and not has_best:
        print('补存峰值权重：按原配置重新训练；保留原末轮权重和历史归档。',flush=True)
        train_missing(run,model,a,A,x,p,device,recover_best=True)
        has_best=True;action='trained_to_save_best'
    elif not final_path.exists() and not has_best:
        if inference_only: raise FileNotFoundError(final_path)
        print(f'Weight absent; training: {run["sample"]} seed {run["seed"]}',flush=True)
        train_missing(run,model,a,A,x,p,device);has_best=True;action='trained_missing_weight'
    kind='best' if model_checkpoint=='best' or (model_checkpoint=='auto' and has_best) else 'final'
    path=best_weight_path(run) if kind=='best' else final_path
    if path.exists():
        if action=='loaded_existing_weight':
            print(f'Load existing weight; skip training: {run["sample"]} {run["profile"]} seed {run["seed"]}',flush=True)
        state=torch.load(path,map_location=device,weights_only=True)
        validate_weight(run,state,kind);model.load_state_dict(state['model'],strict=True)
    else:
        raise FileNotFoundError(f'{path}\n旧实验没有峰值权重；需启用 TRAIN_FOR_BEST 才能重跑补存。')
    if not has_best:
        print('提示：当前只有末轮权重，没有完整训练的峰值权重；本次仅预测，不会自动重跑补存。',flush=True)
    print(f'实际加载：{kind}, epoch={state["epoch"]}',flush=True)
    labels=a.obs['reference'].astype(str).to_numpy()
    z,pred,metrics=evaluate(model,x,p,labels,run['parameters']['clusters'],run['seed'])
    dest=local(run['prediction'])/('best' if kind=='best' else '')
    save_arrays(dest/'prediction.npz',a,z,pred,state['epoch'])
    comparison={}
    archived=local(run['archive'])/('peak.npz' if kind=='best' else 'final.npz')
    if archived.exists():
        with np.load(archived,allow_pickle=False) as old:
            if not np.array_equal(old['obs_names'],a.obs_names.to_numpy(dtype=str)):
                raise ValueError('Archived observation order does not match input')
            comparison=dict(max_abs_embedding_delta=float(np.max(np.abs(z-old['embedding']))),
                clustering_agreement_ARI=float(adjusted_rand_score(old['prediction'],pred)),
                archived_ARI=float(adjusted_rand_score(old['reference'],old['prediction'])))
    report=dict(action=action,device=device,epoch=state['epoch'],checkpoint=kind,
        metrics=metrics,weight=path.relative_to(ROOT).as_posix(),weight_sha256=sha(path),
        best_weight_available=has_best,prediction=dest.relative_to(ROOT).as_posix())
    report['comparison_with_archived_peak' if kind=='best' else 'comparison_with_archived_final']=comparison
    if kind=='best':
        peak_file=local(state['peak_result'])
        report['peak_reload_check']=dict(saved_ARI=state['metrics']['ARI'],
            ARI_difference=metrics['ARI']-state['metrics']['ARI'],prediction_array_available=peak_file.exists())
        if peak_file.exists():
            if sha(peak_file)!=state['peak_result_sha256']: raise ValueError('Saved peak prediction changed')
            with np.load(peak_file,allow_pickle=False) as old:
                report['peak_reload_check'].update(
                    max_abs_embedding_delta=float(np.max(np.abs(z-old['embedding']))),
                    clustering_agreement_ARI=float(adjusted_rand_score(old['prediction'],pred)))
    write(dest/'metrics.json',report)
    pd.DataFrame(dict(obs_name=a.obs_names,prediction=pred,reference=labels)).to_csv(dest/'domains.csv',index=False)
    print(f'ARI={metrics["ARI"]:.6f}; saved to {dest.relative_to(ROOT)}',flush=True)
    del model,x,p
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return report


def summary(print_table=True):
    rows=[]
    for run in configuration()['runs']:
        m=read(local(run['archive'])/'metrics.json')
        rows.append(dict(dataset=run['dataset'],sample=run['sample'],donor=run['donor'],
            profile=run['profile'],seed=run['seed'],peak=m['peak'],final=m['final'],peak_epoch=m['peak_epoch']))
    df=pd.DataFrame(rows);out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'seed_metrics.csv',index=False)
    bysample=df.groupby(['dataset','sample','profile'],sort=False).agg(
        mean_peak=('peak','mean'),sd_peak=('peak','std'),mean_final=('final','mean'),sd_final=('final','std'),seeds=('seed','count'))
    bysample.to_csv(out/'sample_metrics.csv')
    agg=[]
    for (ds,profile),g in df.groupby(['dataset','profile']):
        for metric in ['peak','final']:
            v=g.groupby('sample')[metric].mean().to_numpy() if ds=='DLPFC' else g[metric].to_numpy()
            agg.append(dict(dataset=ds,profile=profile,metric=metric,mean=float(v.mean()),
                median=float(np.median(v)),sd=float(v.std(ddof=1)),n=len(v),
                unit='slice_seed_mean' if ds=='DLPFC' else 'seed'))
    pd.DataFrame(agg).to_csv(out/'dataset_metrics.csv',index=False)
    if print_table: print(bysample.to_string(),flush=True)
    return df


def verify():
    runs=configuration()['runs'];seen=set();count=0;best_count=0;partial_count=0
    for run in runs:
        identity=(run['sample'],run['profile'],run['seed'])
        if identity in seen: raise ValueError('Duplicate run')
        seen.add(identity)
        state=torch.load(local(run['weight']),map_location='cpu',weights_only=True)
        validate_weight(run,state)
        if best_weight_path(run).exists():
            peak=torch.load(best_weight_path(run),map_location='cpu',weights_only=True)
            validate_weight(run,peak,'best',require_complete=False)
            if completed_best(run):
                if sha(local(peak['peak_result']))!=peak['peak_result_sha256']:
                    raise ValueError('Peak checkpoint prediction checksum mismatch')
                best_count+=1
            else: partial_count+=1
        m=read(local(run['archive'])/'metrics.json')
        for f,digest in m['files'].items():
            if sha(local(run['archive'])/f)!=digest: raise ValueError('Archived result modified: '+f)
        h=pd.read_csv(local(run['archive'])/'history.csv')
        if len(h)!=run['parameters']['epochs'] or not np.allclose([h.eval_ARI.max(),h.eval_ARI.iloc[-1]],
            [m['peak'],m['final']],atol=1e-10,rtol=0): raise ValueError('History/metric mismatch')
        for label in ['peak','final']:
            with np.load(local(run['archive'])/f'{label}.npz',allow_pickle=False) as z:
                score=adjusted_rand_score(z['reference'],z['prediction'])
                expected_epoch=m['peak_epoch'] if label=='peak' else m['final_epoch']
                if abs(score-m[label])>1e-10 or int(z['epoch'])!=expected_epoch:
                    raise ValueError('Archived predictions do not reproduce the recorded metric/epoch')
        count+=1
    for run in {r['input']:r for r in runs}.values():
        inputs(run)
    print(f'Verified {count} weights and archived runs, all prepared inputs.',flush=True)
    comparisons=[]
    for run in runs:
        p=local(run['prediction'])/'metrics.json'
        if p.exists():
            result=read(p)
            comparisons.append(dict(sample=run['sample'],profile=run['profile'],seed=run['seed'],
                **result.get('comparison_with_archived_final',{})))
    write(ROOT/'results/verification.json',dict(weights=count,archived_runs=count,passed=True,
        complete_best_weights=best_count,incomplete_best_weights=partial_count,missing_best_weights=count-best_count-partial_count,
        existing_prediction_checks=comparisons))
