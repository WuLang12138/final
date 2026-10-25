"""Notebook-only presentation: stream subprocess logs and display saved evidence.

Does not import training libraries or change models, metrics, or result paths.
"""
from pathlib import Path
import base64
from collections import deque
from html import escape
from io import BytesIO
import json
import os
import subprocess
import sys

import pandas as pd
from IPython.display import HTML, Markdown, display


class NotebookResults:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, path):
        return json.loads((self.root/path).read_text(encoding='utf-8'))

    def records(self, dataset=None, profile=None, samples=None, seeds=None):
        return [r for r in self.read('config/paper.json')['runs']
                if (dataset is None or r['dataset']==dataset)
                and (profile is None or r['profile']==profile)
                and (samples is None or r['sample'] in samples)
                and (seeds is None or r['seed'] in seeds)]

    @staticmethod
    def table(frame, title):
        display(Markdown('**'+title+'**'))
        # DataFrame HTML avoids truncating DLPFC's 60 per-seed rows.
        display(HTML('<div style="overflow-x:auto">'+frame.to_html(
            index=False,escape=True,na_rep='—',float_format=lambda x:f'{x:.6f}')+'</div>'))

    def archived(self, dataset=None, profile=None):
        rows=[]
        for run in self.records(dataset,profile):
            m=self.read(run['archive']+'/metrics.json')
            rows.append(dict(dataset=run['dataset'],sample=run['sample'],profile=run['profile'],
                             seed=run['seed'],peak=m['peak'],final=m['final']))
        if not rows:
            display(Markdown('没有符合筛选条件的归档结果。'));return
        df=pd.DataFrame(rows)
        bysample=df.groupby(['dataset','sample','profile'],sort=False).agg(
            n=('seed','count'),peak_mean=('peak','mean'),peak_sd=('peak','std'),
            final_mean=('final','mean'),final_sd=('final','std')).reset_index()
        self.table(bysample.rename(columns=dict(dataset='数据',sample='样本',profile='方案',n='seed数',
            peak_mean='归档峰值ARI均值',peak_sd='峰值ARI标准差',
            final_mean='归档末轮ARI均值',final_sd='末轮ARI标准差')),'历史归档：逐样本汇总')
        aggregate=[]
        for (ds,pr),g in df.groupby(['dataset','profile'],sort=False):
            for key,title in [('peak','归档峰值'),('final','归档末轮')]:
                v=g.groupby('sample')[key].mean() if ds=='DLPFC' else g[key]
                aggregate.append({'数据':ds,'方案':pr,'结果类型':title,'统计单位':
                    '切片的seed均值' if ds=='DLPFC' else 'seed','数量':len(v),
                    'ARI均值':v.mean(),'ARI中位数':v.median(),'ARI标准差':v.std(ddof=1)})
        self.table(pd.DataFrame(aggregate),'历史归档：数据集汇总')
        display(Markdown('峰值是各 seed 训练过程中最高 ARI，**不是当前末轮权重重新预测的结果**。'
            'DLPFC 先对每片的 seed 取均值，再汇总切片；标准差均为样本标准差。'))

    def predictions(self, dataset=None, profile='paper', samples=None, seeds=None, executions=None):
        rows=[];loaded=trained=0
        actions={'loaded_existing_weight':'加载已有权重，跳过训练',
                 'trained_missing_weight':'权重缺失，完成训练','trained_to_save_best':'完成重跑，补存峰值权重'}
        selected=self.records(dataset,profile,samples,seeds)
        if executions is not None:
            indexed={(r['sample'],r['profile'],r['seed']):r for r in executions}
            if set(indexed)!={(r['sample'],r['profile'],r['seed']) for r in selected}:
                raise RuntimeError('执行报告与本次选择不一致，停止展示以免混用结果。')
        for run in selected:
            r=(indexed[(run['sample'],run['profile'],run['seed'])] if executions is not None
               else self.read(run['prediction']+'/metrics.json'))
            a=self.read(run['archive']+'/metrics.json')
            is_best=r.get('checkpoint')=='best'
            comp=r.get('comparison_with_archived_peak' if is_best else 'comparison_with_archived_final',{})
            reload=r.get('peak_reload_check',{})
            action=r['action'];loaded+=action=='loaded_existing_weight';trained+=action.startswith('trained_')
            rows.append({'数据':run['dataset'],'样本':run['sample'],'方案':run['profile'],'seed':run['seed'],
                '执行方式':actions[action],'加载权重类型':'峰值模型' if is_best else '末轮模型',
                '完整峰值权重':'已保存' if r.get('best_weight_available') else '未保存；需重跑补存',
                '权重轮次':r['epoch'],'本次预测ARI':r['metrics']['ARI'],'本次预测NMI':r['metrics']['NMI'],
                '历史对照类型':'归档峰值' if is_best else '归档末轮',
                '历史对照ARI':a['peak'] if is_best else a['final'],
                '与历史对照ARI差值':r['metrics']['ARI']-(a['peak'] if is_best else a['final']),
                '表示最大绝对差':comp.get('max_abs_embedding_delta'),
                '聚类一致性ARI':comp.get('clustering_agreement_ARI'),
                '与新存峰值ARI差值':reload.get('ARI_difference'),
                '归档峰值ARI（参考）':a['peak']})
        if not rows: raise ValueError('没有匹配的本次预测结果')
        display(Markdown(f'**本次完成 {len(rows)} 个实验：加载已有模型 {loaded} 个，本次训练 {trained} 个。**'))
        frame=pd.DataFrame(rows);self.table(frame,'本次预测：逐 seed 指标、实际加载轮次与历史对照')
        grouped=frame.groupby(['数据','样本','方案','加载权重类型'],sort=False).agg(
            seed数=('seed','count'),ARI均值=('本次预测ARI','mean'),ARI标准差=('本次预测ARI','std'),
            NMI均值=('本次预测NMI','mean'),NMI标准差=('本次预测NMI','std')).reset_index()
        self.table(grouped,'本次预测：逐样本均值与标准差')
        if dataset=='DLPFC' and frame['加载权重类型'].nunique()==1:
            values=grouped['ARI均值']
            self.table(pd.DataFrame([{'切片数':len(values),'切片均值的平均数':values.mean(),
                '切片均值的中位数':values.median(),'切片间标准差':values.std(ddof=1)}]),'本次预测：DLPFC 切片汇总')
        display(Markdown('本次使用的模型以“加载权重类型”和“权重轮次”为准；不同类型分开汇总。'
            '重跑获得的峰值可能与历史归档不同。“与新存峰值ARI差值”用于核对峰值权重重新加载后能否恢复新训练的峰值。'
            '聚类一致性 ARI=1 表示分区一致，表示最大绝对差=0 表示表示数值完全一致。'))

    def current(self):
        out=self.root/'results/tables/current'
        report=self.read('results/tables/current/source_manifest.json')
        selected=pd.DataFrame([{'数据':ds,'方案':v['profile'],'权重类型':v['checkpoint']}
            for ds,v in report['selection']['datasets'].items()
            if ds in {r['dataset'] for r in report['runs']}])
        self.table(selected,'当前读取方案（最近成功运行的实验）')
        for filename,title in [('seed_metrics.csv','当前预测：逐 seed 结果'),
                               ('sample_metrics.csv','当前预测：逐样本汇总'),
                               ('dataset_metrics.csv','当前预测：数据集汇总')]:
            df=pd.read_csv(out/filename,dtype={'sample':str})
            if filename=='seed_metrics.csv':
                df=df[['dataset','sample','profile','seed','checkpoint','epoch','ARI','NMI','reload_ARI_delta']]
            self.table(df.rename(columns={'dataset':'数据','sample':'样本','profile':'方案','seed':'seed',
                'checkpoint':'权重类型','epoch':'轮次','seeds':'seed数','mean_ARI':'ARI均值','sd_ARI':'ARI标准差',
                'median_ARI':'ARI中位数','mean_NMI':'NMI均值','sd_NMI':'NMI标准差',
                'n':'统计单位数','unit':'统计单位','reload_ARI_delta':'与新训练峰值ARI差值'}),title)
        display(Markdown('以上指标由当前模型预测文件中的标签重新计算，**未读取历史归档补齐结果**。'
            'DLPFC 先计算每张切片的 seed 均值，再计算切片间均值、中位数及样本标准差。'
            '这里不包含库外的 GCN／其他方法对照或消融实验。'))

    def verification(self, source='current'):
        if source=='current':
            r=self.read('results/current_verification.json')
            self.table(pd.DataFrame([{'核验结果':'通过' if r['passed'] else '未通过',
                '应有实验数':r['expected_runs'],'核验通过数':r['validated_runs'],
                '历史归档补齐':'没有','缺失或异常数':len(r['errors'])}]),'当前预测、模型及新训练记录核验')
            display(Markdown('核对模型配置及校验值、预测标签重算指标、输入点位顺序和完整训练记录；不训练、不重新预测。'))
            return
        r=self.read('results/verification.json')
        self.table(pd.DataFrame([{'核验结果':'通过' if r['passed'] else '未通过',
            '末轮权重数':r['weights'],'完整峰值权重数':r.get('complete_best_weights','尚未核验'),
            '未完成训练的峰值权重数':r.get('incomplete_best_weights','尚未核验'),
            '缺失峰值权重数':r.get('missing_best_weights','尚未核验'),
            '归档实验数':r['archived_runs']}]),'文件、配置及归档指标核验')
        display(Markdown('该步骤检查权重配置、输入和归档指标；**不会训练或重新预测**。'))

    def figures(self, checkpoint='current', dataset=None, samples=None, profile=None):
        from PIL import Image
        names={'current':'当前实验预测','peak':'归档峰值','final':'归档末轮','prediction':'已有预测文件'}
        display(Markdown(f'**图件预览：{names[checkpoint]}。** 下方直接显示全部已生成图件；可点击样本名称折叠，高清文件照常保存。'))
        groups=[('模型框架',[self.root/'results/figures/framework/Fig1_SpaSRD_framework.png'])]
        if checkpoint=='current':
            runs=self.read('results/figures/current/source_manifest.json')['runs']
        else:runs=self.records(dataset,profile or 'paper',samples)
        for ds,pr in dict.fromkeys((r['dataset'],r['profile']) for r in runs):
            folder=self.root/'results/figures'/checkpoint/ds/pr
            groups.append((ds+' / '+pr+'：seed 箱线图',[folder/'seed_boxplot.png']))
            for sample in dict.fromkeys(r['sample'] for r in runs if r['dataset']==ds and r['profile']==pr):
                groups.append((ds+' / '+pr+' / '+sample,[folder/sample/(kind+'.png') for kind in ['spatial','confusion','umap']]))
        shown=0
        for title,paths in groups:
            pictures=[]
            for path in paths:
                if not path.exists():
                    pictures.append('<p>未找到已生成图件：'+escape(path.name)+'</p>');continue
                # Embed small previews, not full-resolution images, to keep notebooks light.
                with Image.open(path) as img:
                    img=img.convert('RGB');img.thumbnail((1000,1000))
                    buf=BytesIO();img.save(buf,format='JPEG',quality=88)
                encoded=base64.b64encode(buf.getvalue()).decode('ascii')
                pictures.append('<img alt="'+escape(path.stem)+'" style="max-width:100%;height:auto" '
                                'src="data:image/jpeg;base64,'+encoded+'"/>')
                shown+=1
            display(HTML('<details open><summary><b>'+escape(title)+
                         '</b></summary>'+''.join(pictures)+'</details>'))
        print(f'Notebook 内已嵌入 {shown} 张预览图。')

    @staticmethod
    def option(args, key, default=None, multiple=False):
        if key not in args:return default
        values=[]
        for item in args[args.index(key)+1:]:
            if item.startswith('--'):break
            values.append(item)
        return values if multiple else values[0]

    def execute(self, *arguments):
        args=list(map(str,arguments))
        command=[sys.executable,'-u',str(self.root/'process/run.py'),*args,'--notebook']
        print('开始执行：'+' '.join(args),flush=True)
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        tail=deque(maxlen=30)
        # Capture then print each line into IPython; inherited subprocess stdout
        # may otherwise be routed to a VS Code terminal instead of the cell.
        with subprocess.Popen(command,cwd=self.root,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    tail.append(line);print(line,end='',flush=True)
                code=proc.wait()
            except BaseException:
                proc.terminate();proc.wait();raise
        if code:
            raise RuntimeError('运行失败；上方已显示日志。本次不展示旧文件作为新结果。\n'+''.join(tail))
        action=args[0];dataset=self.option(args,'--dataset');profile=self.option(args,'--profile','paper')
        samples=self.option(args,'--sample',multiple=True);seeds=self.option(args,'--seed',multiple=True)
        if action=='run':self.predictions(dataset,profile,samples,None if seeds is None else list(map(int,seeds)),
                                          self.read('results/last_execution.json'))
        elif action=='verify':self.verification(self.option(args,'--source','current'))
        elif action=='summary':
            if self.option(args,'--source','current')=='current':self.verification();self.current()
            else:self.archived(dataset,self.option(args,'--profile'))
        elif action=='figures':self.figures(self.option(args,'--checkpoint','current'),dataset,samples,self.option(args,'--profile'))
