"""Main-method panels: spatial fidelity and five-seed variation, no invented controls.

Python quantitative grid; vector PDF/SVG plus PNG preview and underlying CSV.
Archived label-assisted peaks and fixed-budget predictions are never pooled.
"""
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from . import workflow as w


def save(fig, stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','svg','png']:
        fig.savefig(stem.with_suffix('.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)


def align(reference, prediction):
    labels=sorted(np.unique(reference)); pred_labels=sorted(np.unique(prediction))
    table=np.array([[np.sum((reference==a)&(prediction==b)) for b in pred_labels] for a in labels])
    ri,ci=linear_sum_assignment(-table)
    mapping={pred_labels[j]:labels[i] for i,j in zip(ri,ci)}
    return labels,np.array([mapping.get(p,'Unmatched '+str(p)) for p in prediction])


def generate(checkpoint='current',dataset=None,samples=None,profile=None):
    from umap import UMAP
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    draw_framework(w.ROOT/'results/figures/framework')
    config=w.configuration();seed=config['plotting']['seed']
    current_rows=None
    if checkpoint=='current':
        from .current_results import collect
        current_rows,source_manifest=collect(dataset,profile,samples)
        selected={(r['sample'],r['profile'],r['seed']):r for r in current_rows.to_dict('records')}
        runs=[r for r in config['runs'] if (r['sample'],r['profile'],r['seed']) in selected]
    else:
        runs=[r for r in config['runs'] if r['profile']==(profile or 'paper')
              and (dataset is None or r['dataset']==dataset) and (not samples or r['sample'] in samples)]
    out=w.ROOT/'results/figures'/checkpoint
    rows=[]
    for run in runs:
        if checkpoint=='current':
            source=selected[(run['sample'],run['profile'],run['seed'])]
            file=w.local(source['prediction'])
        else:
            prediction_dir=w.local(run['prediction'])/('best' if checkpoint=='prediction' and w.completed_best(run) else '')
            file=(prediction_dir/'prediction.npz') if checkpoint=='prediction' else (w.local(run['archive'])/f'{checkpoint}.npz')
        if not file.exists():
            raise FileNotFoundError(f'Run the corresponding dataset notebook first: {file}')
        with np.load(file,allow_pickle=False) as obj:
            z={k:obj[k] for k in obj.files}
        ref=z['reference'].astype(str); pred=z['prediction'].astype(str)
        ari=float(adjusted_rand_score(ref,pred))
        rows.append(dict(dataset=run['dataset'],sample=run['sample'],profile=run['profile'],seed=run['seed'],ARI=ari,
                         epoch=int(z['epoch']),source=file.relative_to(w.ROOT).as_posix(),source_sha256=w.sha(file)))
        if run['seed']!=seed: continue
        print(f'Drawing {run["sample"]}, {run["profile"]}, {checkpoint}, seed {seed}',flush=True)
        labels,aligned=align(ref,pred);all_labels=labels+sorted(set(aligned)-set(labels))
        palette=plt.get_cmap('tab20',max(20,len(all_labels)))
        colors={lab:palette(i) for i,lab in enumerate(all_labels)}
        xy=z['spatial'];fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
        for ax,v,title in zip(axes,[ref,aligned],['Reference',f'SpaSRD: ARI = {ari:.3f}']):
            ax.scatter(xy[:,0],xy[:,1],c=[colors[t] for t in v],s=2,rasterized=True)
            ax.set_aspect('equal');ax.invert_yaxis();ax.set_axis_off();ax.set_title(title)
        handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[l],markersize=3,label=l) for l in all_labels]
        fig.legend(handles=handles,loc='outside lower center',ncol=min(5,len(all_labels)),fontsize=6)
        fig.suptitle(f'{run["sample"]} | {run["profile"]} | {checkpoint} | epoch {int(z["epoch"])} | seed {seed}')
        dest=out/run['dataset']/run['profile']/run['sample'];save(fig,dest/'spatial')
        cm=confusion_matrix(ref,aligned,labels=all_labels).astype(float)
        cm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)!=0)
        fig,ax=plt.subplots(figsize=(4.2,3.7),layout='constrained')
        im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax,label='Fraction within reference region')
        ax.set_xticks(range(len(all_labels)),all_labels,rotation=90,fontsize=6)
        ax.set_yticks(range(len(all_labels)),all_labels,fontsize=6)
        ax.set_xlabel('SpaSRD region (label-matched for display)');ax.set_ylabel('Reference region')
        save(fig,dest/'confusion')
        pd.DataFrame(cm,index=all_labels,columns=all_labels).to_csv(dest/'confusion.csv')
        low=UMAP(n_components=2,n_neighbors=config['plotting']['umap_neighbors'],
            min_dist=config['plotting']['umap_min_dist'],random_state=0,n_jobs=1).fit_transform(z['embedding'])
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
        for ax,v,title in zip(axes,[ref,aligned],['UMAP: reference','UMAP: SpaSRD']):
            ax.scatter(low[:,0],low[:,1],c=[colors[t] for t in v],s=2,rasterized=True)
            ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2')
        fig.legend(handles=handles,loc='outside lower center',ncol=min(5,len(all_labels)),fontsize=6)
        save(fig,dest/'umap')
        pd.DataFrame(dict(obs_name=z['obs_names'],x=xy[:,0],y=xy[:,1],umap1=low[:,0],umap2=low[:,1],
            reference=ref,prediction=pred,display_label=aligned)).to_csv(dest/'plot_source.csv',index=False)
    if not rows: raise ValueError('No selected figure inputs')
    df=pd.DataFrame(rows);out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'seed_metrics.csv',index=False)
    for (ds,pr),g in df.groupby(['dataset','profile'],sort=False):
        order=list(g['sample'].unique());fig,ax=plt.subplots(figsize=(max(3.2,len(order)*.5),3),layout='constrained')
        values=[g.loc[g['sample']==sample,'ARI'].to_numpy() for sample in order]
        ax.boxplot(values,tick_labels=order,showfliers=False,widths=.5)
        for i,v in enumerate(values,1):
            ax.scatter(i+np.linspace(-.09,.09,len(v)),v,s=14,color='#4678A4',zorder=3)
        ax.tick_params(axis='x',rotation=45);ax.set_ylabel('ARI');ax.set_ylim(0,1)
        ax.set_title(f'{ds} / {pr}: {checkpoint}; each dot is one seed')
        save(fig,out/ds/pr/'seed_boxplot')
    w.write(out/'figure_provenance.json',dict(checkpoint=checkpoint,seed_for_panels=seed,
        evidence='Spatial agreement and per-seed variation of the main method; not a baseline comparison.',
        peak_is_label_assisted=checkpoint=='peak' or (checkpoint=='current' and 'best' in set(current_rows.checkpoint)),
        sources=rows,archive_fallback=False,
        umap='descriptive projection, not an independent accuracy test',
        alignment='Hungarian label matching is used only for display; ARI uses original labels.'))
    if checkpoint=='current':w.write(out/'source_manifest.json',source_manifest)
    print('Figures:',out,flush=True)


def draw_framework(output):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42});output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.2,7.0));ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    blue='#4C78A8';orange='#BE7844';green='#4C8C82'
    def txt(x,y,s,size=7,**kw):ax.text(x,y,s,fontsize=size,ha='center',va='center',**kw)
    def box(x,y,w,h,s,c='#667680',fill='#F4F6F7',size=7):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35,rounding_size=1.2',lw=.8,ec=c,fc=fill));txt(x+w/2,y+h/2,s,size)
    def arrow(a,b,c='#555555',style='->',rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=9,lw=.9,color=c,connectionstyle=f'arc3,rad={rad}'))
    def title(y,letter,s):
        ax.text(0,y,letter,fontsize=10,fontweight='bold');ax.text(4,y,s,fontsize=8.5,fontweight='bold')
    title(98,'a','Shared spatial transcriptomic input')
    box(1,85,25,8,'Expression matrix X\nSingle molecular modality')
    box(34,85,24,8,'Spatial coordinates\n8 nearest neighbours')
    box(67,85,31,8,'Symmetric spatial graph A\nP = D⁻¹(A + I)')
    arrow((58.5,89),(66.5,89));txt(50,81,'Choose one route for each experimental configuration',7,color='#444444')
    title(76,'b','SpaSRD-D   Restarted diffusion with a fixed feature anchor')
    box(1,61,20,10,'X → MLP\nInitial features H₀',blue,'#EFF4F8')
    box(29,61,45,10,'Hₖ = (1 − α) P Hₖ₋₁ + αH₀\nK propagation steps',blue,'#EFF4F8',8)
    box(82,61,16,10,'Projection\nZᴰ',blue,'#EFF4F8')
    arrow((21.5,66),(28.5,66),blue);arrow((74.5,66),(81.5,66),blue)
    arrow((11,71.5),(51,71.5),blue,rad=-.21);txt(31,74,'restart anchor',6,color=blue)
    txt(50,57,'Spatial context is integrated without stacking K trainable graph layers.',7)
    title(52,'c','SpaSRD-S   Separate common context from local differences')
    box(1,34,17,10,'X → MLP\nH',green,'#F0F6F4')
    box(27,42,29,8,'Common channel\nC = LN(Linear(PH))',green,'#F0F6F4')
    box(27,27,29,10,'Differential channel\nR = LN(Linear(H − PH) + Linear(H))',orange,'#FAF3ED',6)
    box(66,34,32,10,'Gate G = σ(Linear([C, R]))\n[C, G ⊙ R] → projection → Zˢ',green,'#F0F6F4',7)
    arrow((18.5,41),(26.5,46),green);arrow((18.5,37),(26.5,32),orange)
    arrow((56.5,46),(65.5,41),green);arrow((56.5,32),(65.5,37),orange)
    txt(50,23,'The graph constrains C; both channels contribute to the clustering representation.',7)
    title(18,'d','Shared reconstruction objective and domain assignment')
    box(1,5,31,9,'Expression reconstruction\nfrom Zᴰ or Zˢ',fill='#F6F6F6')
    box(37,5,27,9,'Graph reconstruction\nfrom Zᴰ (D) or C (S)',fill='#F6F6F6')
    box(70,5,28,9,'K-means on Zᴰ or Zˢ\nSpatial domains',fill='#F6F6F6')
    txt(32,1,'Training objective: L = Lexpr + β Lgraph',7)
    txt(84,1,'Clustering after training',6.5)
    fig.subplots_adjust(left=.025,right=.985,top=.985,bottom=.02)
    for ext in ['svg','pdf','png']:fig.savefig(output/f'Fig1_SpaSRD_framework.{ext}',dpi=300,facecolor='white')
    w.write(output/'Fig1_SpaSRD_framework.json',dict(
        kind='architecture schematic, not measured data',
        source_modules={p:w.sha(w.ROOT/p) for p in ['model/propagation.py','model/acm.py']},
        script_sha256=w.sha(__file__),route_selection='Experiment configuration; no automatic router or joint fusion',
        common_channel='Graph loss uses the preprojection common channel in SpaSRD-S',
        omissions='Bias terms, dropout, and detailed expression decoder heads omitted for clarity'))
    plt.close(fig);return output/'Fig1_SpaSRD_framework.png'
