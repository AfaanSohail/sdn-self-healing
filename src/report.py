"""Build the report exclusively from executed CSV/JSON outputs and chart files."""
from pathlib import Path
import json,subprocess,shutil,os
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

B=Path(__file__).resolve().parents[1];R=B/'results'
D=Document();sec=D.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.75)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 s=D.styles[name];s.font.name='Times New Roman';s.font.color.rgb=RGBColor(0,0,0)
 s.font.size=Pt(10.5 if name in ('Title','Heading 1','Heading 2') else 9)
 s.paragraph_format.space_after=Pt(6)
 s.paragraph_format.line_spacing=1.5
D.styles['Normal'].paragraph_format.line_spacing=1.5
D.styles['Title'].paragraph_format.space_after=Pt(12)

def p(text,style=None):return D.add_paragraph(text,style)
def h(text):D.add_heading(text,1)
def page(title):
 heading=D.add_heading(title,1);heading.paragraph_format.page_break_before=True
def table(headers,rows,widths=None):
 t=D.add_table(rows=1,cols=len(headers));t.autofit=False;t.style='Table Grid'
 if widths:
  for i,w in enumerate(widths):t.columns[i].width=Inches(w)
 for i,z in enumerate(headers):t.rows[0].cells[i].text=str(z)
 for row in rows:
  cells=t.add_row().cells
  for i,z in enumerate(row):cells[i].text=str(z)
 for ri,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr();prevent=OxmlElement('w:cantSplit');trpr.append(prevent)
  if ri==0:
   repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
  for ci,c in enumerate(row.cells):
   if widths:c.width=Inches(widths[ci])
   c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   tcpr=c._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
   for side in ('top','left','bottom','right'):
    e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'000000');borders.append(e)
   tcpr.append(borders)
   if ri==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'FFFFFF');tcpr.append(shade)
   for par in c.paragraphs:
    par.paragraph_format.space_after=Pt(3);par.paragraph_format.space_before=Pt(3)
    if ci>0 and len(str(c.text))<35:par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for run in par.runs:run.font.size=Pt(9);run.bold=ri==0
 return t

def fig(name,caption,width=6.7):
 D.add_picture(str(R/'figures'/f'{name}.png'),width=Inches(width))
 D.paragraphs[-1].paragraph_format.space_before=Pt(6);D.paragraphs[-1].paragraph_format.space_after=Pt(0)
 p(caption,'Caption')
def fmt(x,n=3):return 'N/A' if x is None or pd.isna(x) else f'{x:.{n}f}'
def percent(x):return fmt(100*x,2)
def short(m):return {'B0_no_recovery':'B0 No remediation','B1_rule':'B1 Shortest path','B2_greedy':'B2 Greedy','B3_GA':'B3 GA','B4_PSO':'B4 PSO','Full_DQN':'Full DQN','A1_no_diagnosis':'A1 No diagnosis','A2_no_shield':'A2 No shield','A4_predictive':'A4 Predictive','A5_no_topology':'A5 No topology','A6_no_SHAP':'A6 No SHAP','Oracle':'Oracle diagnosis'}[m]

if __name__=='__main__':
 summary=json.loads((R/'summary.json').read_text());lookup={(v['topology'],v['method']):v for v in summary}
 det=json.loads((R/'detection_metrics.json').read_text());pub=json.loads((R/'spotlight_metrics.json').read_text());stats=json.loads((R/'paired_statistics.json').read_text());audit=json.loads((R/'evaluation_audit.json').read_text());validation=json.loads((R/'validation.json').read_text())
 df=pd.read_csv(R/'episodes.csv');primary=df[(df.feasible==1)&(df.scenario!='switch')];full=lookup[(12,'Full_DQN')]
 core=['B0_no_recovery','B1_rule','B2_greedy','B3_GA','B4_PSO','Full_DQN']
 p('Safety constrained SDN self healing','Title')
 p('Phase 3 Research Implementation and Experimental Evaluation','Subtitle')
 p('Computer Networks Research Project\nStudent IDs 24i0081 and 24i0005\n15 November 2026')
 h('Study scope and main finding')
 p('This project implements and evaluates a connected fault detection, diagnosis, recovery and verification pipeline. The study asks whether learned recovery improves protected services under diverse faults while limiting collateral disruption. We revised the supplied deterministic pilot and executed a stochastic simulation study with matched baselines, component ablations and an external detection check.')
 p(f'The final evaluation contains {len(df):,} reset episodes across two topologies, 11 scenario families, three loads and 12 methods. Twenty independently seeded DQN policies were trained for 50,000 environment steps each. The 30 environmental blocks per stratum include all five model seeds, six blocks per seed. All results below are regenerated from the delivered outputs.')
 best=min((lookup[(12,m)] for m in core[1:]),key=lambda r:r['sla_burden'])
 p(f'On feasible known faults in the 12-switch topology, Full DQN had {percent(full["sla_burden"])}% mean SLA burden and {fmt(full["capped_unrecovered_seconds"],2)} seconds capped time unrecovered. The lowest measured SLA burden among the core methods was {short(best["method"])} at {percent(best["sla_burden"])}%. These observations are interpreted with paired intervals and training-seed sensitivity, not as an assumed advantage for reinforcement learning.')
 p(f'The official SpotLight platform sample produced LightGBM F1={fmt(pub["metrics"]["LightGBM"]["F1"])}. This small external result is more difficult than the synthetic test and does not reproduce RANGAN\'s full-dataset benchmark.')
 h('Evidence boundary')
 p('The implementation is a Python service-KPI and routing simulator. It does not run Mininet, Open vSwitch, an OpenFlow controller, real packet traffic or physical devices. Its acknowledgements are simulated route-state updates. The external benchmark uses one official public sample trace. The report documents these changes from Phase 2 and does not claim production SDN validation, full public-dataset replication or guaranteed safety.')

 page('1 Changes from Phase 2 and review response')
 p('The teacher gave no specific feedback in the preceding class, as confirmed by the student. The supplied package review is treated as revision feedback rather than attributed to the teacher. The architecture and research questions remain those of Phase 2; the experimental platform is a documented substitution.')
 table(['Issue or design commitment','Executed revision or limitation'],[
 ('Deterministic three-block pilot','30 matched environmental blocks per scenario/load/topology; randomized strengths, locations, traffic and telemetry gaps.'),
 ('1,800 DQN steps; incomplete seed coverage','50,000 steps for every Full/A1/A2/A5 seed; five seeds; validation every 10,000 steps and selected/final weights retained.'),
 ('B1 and B2 overlap; B3/B4 absent','Different hop- and delay-aware objectives; heterogeneous link delays and capacity stress; bounded local GA and PSO added.'),
 ('Empty shield ablation and harm counter','Proposal-before-shield instrumentation; stale/capacity/cooldown rejection; same-traffic counterfactual harm measurement and rollback.'),
 ('Missing ablations and explanations','A4 predictive triggering, A5 retrained topology removal, A6 native TreeSHAP timing and diagnostic oracle; A3 shares B2.'),
 ('Detector gated behind SLA violations','Learned detection runs each second without an SLA prerequisite; B1 alone retains an SLA threshold trigger.'),
 ('Weak external comparison','Official SpotLight platform sample and matched detector baselines; comparison remains limited by sample/task differences.'),
 ('Missing plots and weak inference','ROC/PR, confusion matrices, learning curves, recovery CDF, safety/speed, error bars, per-seed tables and Friedman without B0.'),
 ('Mininet and 500,000-step upper bound','Simulation retained; 50,000-step cap is below the Phase 2 upper bound. No physical-controller or hardware measurements claimed.')],[2.1,4.8])
 p('Additional deviations are retained explicitly: no probability calibration, uncalibrated unknown flags, stale carry-forward rather than median fallback, and a single-sample counterfactual harm criterion rather than the original two-sample collateral-only rule. Proposal selection occurs before shielding so rejection is observable; rejected proposals receive their actual no-op consequence in training. These are part of the implemented protocol, not hidden reproductions of Phase 2.')

 page('2 Data and experimental setup')
 table(['Setting','Executed value'],[
 ('Topologies','12-switch training/base mesh; held-out 20-switch mesh; two host attachments per switch as metadata.'),
 ('Protected traffic','Three flows, 9/11/13 Mbps; 100 Mbps nominal links; randomized nominal one-way delay 1-3 ms.'),
 ('Episode','180 one-second observations; injection 60; clearance 120; secondary cascading fault 70.'),
 ('Workloads','Low/medium/high background means 27/53/72 Mbps per link plus noise and protected traffic.'),
 ('Fault families','Normal, benign burst, outage, congestion, loss, delay, concurrent, correlated, cascading, unseen switch outage and infeasible disconnection.'),
 ('Development split','700 episodes: 60 train, 20 validation, 20 test for each of seven development groups; 20 sampled windows per episode.'),
 ('Recovery test','30 blocks x 11 scenarios x 3 loads x 2 topologies x 12 methods = 23,760 episodes.'),
 ('Seed schedules','Training seeds 11/22/33/44/55; separate supervised, rollout, validation and final-test seed ranges.'),
 ('Thresholds','RTT <=20 ms; loss <=1%; goodput >=90% of demand; five healthy samples for stable recovery.')],[1.45,5.45])
 p('Exogenous traffic noise, missingness and measurement perturbations are indexed by episode seed and time, independently of policy actions. Every method in a paired block receives the same schedule. Methods execute sequentially in seeded random order within a block; eight blocks can run concurrently. Runtime overhead therefore reflects a shared CPU host rather than isolated hardware.')
 p('Preprocessing differences byte counters across received samples and divides by elapsed time. A counter decrease resets rate differencing. Random missing samples and a four-second gap for selected seeds exercise observation age and missingness inputs. Stale values remain explicitly marked, and the shield rejects candidate paths requiring data older than two seconds. This age-gated carry-forward differs from the planned median fallback. The exported preprocessing trace contains raw counters, derived rates, reset markers, missingness and age.')
 p('Ten-sample trailing summaries include mean, maximum, standard deviation and trend. The detector uses 26 global features; component diagnosis uses 23 shared per-link features including neighbour statistics and age/missingness. No future sample enters a feature. Component identifiers join records but are not learned categories. Switch failures and the larger topology are excluded from model training and tuning.')

 baseline_pre=df[(df.method=='B0_no_recovery')&(df.topology==12)&df.scenario.isin(['outage','congestion','loss','delay','concurrent','correlated','cascading'])].clean_sla_burden.mean()
 p(f'Passive pre-injection burden on the base known-fault set was {100*baseline_pre:.3f}%. This interval includes the congestion precursor at seconds 50-59. Background-load violations can also occur without an injected fault. The supplementary healthy-preinjection sensitivity retains entire matched blocks with zero passive burden; it is exploratory and removes some strong precursor cases, so it does not replace the primary analysis.')


 page('3 Pipeline policies and outcome definitions')
 p('A binary LightGBM detector runs unconditionally each second. Two positive samples trigger intervention, with an immediate observed-link-down plus SLA override. Four one-versus-rest link heads estimate outage, congestion, loss and delay; their scores supply class context and candidate component risk. The selected flow has the largest weighted service deficit. Up to three observed-topology paths provide a fixed-size action representation.')
 p('The DQN has a 30-element state, hidden layers of 128 and 64 ReLU units and five actions: no-op, three candidate paths and rollback. Training uses Adam 0.001, discount 0.99, replay capacity 50,000, batch size 64, updates every fourth environment step and target refresh every 500 gradient updates. Epsilon decays from 1 to 0.05 over the first 70% of the 50,000-step budget. Each seed selects the validation checkpoint minimizing harm count first and SLA burden second; earlier ties are retained.')
 table(['Method','Decision mechanism'],[
 ('B0','Passive monitoring without route remediation.'),('B1','SLA trigger; shortest-hop alternate path with deterministic ties.'),('B2 / A3','Shared detector/diagnosis; lowest measured path delay, loss and utilization proxy.'),('B3 and B4','Shared detector/diagnosis; GA/PSO over permitted one-flow candidate actions; population/particles 20, iterations <=20, wall limit 200 ms.'),('Full','Learned detection/diagnosis, DQN proposal, shield, execution and verifier.')],[1.05,5.85])
 p('All deployable methods share candidate construction and at most one flow action per second. The shield checks observed connectivity, endpoint/loop validity, management reachability, telemetry age, residual capacity of at least 1.2 times flow demand and a five-second cooldown. The no-shield ablation retains the experimental management interlock and basic controller validity guards. Diagnosis and explanation outputs do not certify physical causality.')
 p('SLA burden is violating protected-flow seconds divided by all protected-flow seconds during [60,120). Stable recovery begins at the first five-sample healthy interval starting before clearance. Capped time unrecovered assigns 60 seconds to a failure and retains failures in the denominator. Conditional median/p95 describe recovered episodes only; the CDF retains unrecovered episodes. Feasibility checks structural endpoint connectivity at seconds 65 and 80; capacity insufficiency can still cause valid recovery failure.')
 p('The verifier compares the next observation after a route install with keeping the prior route under the identical exogenous traffic. A new collateral violation, >2 percentage-point loss increase or >5 ms RTT increase is counted as harm and initiates rollback. This is a simulation counterfactual and a disclosed alternative harm definition, not a real-network safety guarantee.')

 page('4 Detection and component diagnosis')
 p('The synthetic test has 2,800 feature windows from 140 episode-disjoint test runs. Overlapping windows are not independent statistical replications. LightGBM uses 100 trees, 15 leaves, learning rate 0.05 and balanced class weighting. RF and LightGBM use unscaled data; SVM and normal-only Isolation Forest use standardization. Thresholds are validation-only. The Isolation Forest cutoff now maximizes validation F1 rather than using an insensitive quiet-tail cutoff; the resulting false positives remain reported.')
 table(['Detector','Precision','Recall','F1','ROC AUC','PR AUC'],[[name,*[fmt(v[k]) for k in ('precision','recall','F1','ROC_AUC','PR_AUC')]] for name,v in det['binary_detection'].items()],[2.0,.9,.8,.8,.85,.85])
 fig('synthetic_curves','Figure 1 Synthetic test ROC and precision-recall curves',6.6)
 p(f'Per-link diagnosis macro F1 is {fmt(det["diagnosis"]["macro_F1"])}; top-one component localisation is {fmt(det["diagnosis"]["component_top1_accuracy"])} and top-three is {fmt(det["diagnosis"]["component_top3_accuracy"])}. Concurrent-label Jaccard is {fmt(det["diagnosis"]["concurrent_label_jaccard"])}. These remain easy formula-based faults despite stochastic perturbations. They validate implementation consistency, not general real-network diagnosis. Operational t=65 localisation, conditioned and unconditioned on triggering, is separately exported for recovery runs.')

 page('5 External SpotLight sample detection study')
 p('We used the official SpotLight repository platform.csv and label files [S1]. The source contains one repeated header, which was excluded. Labels are merged as timestamp intervals; IDs, timestamps and labels never enter model features. The retained interval ends 30 seconds after the final annotated event, a fixed task-scope choice before fitting. The benchmark uses numeric platform KPIs, five-sample trailing means, training-fitted median imputation with missingness indicators and model-appropriate scaling.')
 table(['Partition','Rows','Anomalous rows'],[[k,v['rows'],v['positive']] for k,v in pub['split_counts'].items()],[2.2,2.2,2.5])
 p('The split is chronological 60/20/20 with ten-row gaps. It contains one trace rather than independent sessions; adjacent labels and platform conditions remain dependent. No confidence interval treating rows as independent is presented. Validation F1 selects the cutoff for each detector, then the test partition remains held out.')
 table(['Detector','Precision','Recall','F1','ROC AUC'],[[k,*[fmt(v[z]) for z in ('precision','recall','F1','ROC_AUC')]] for k,v in pub['metrics'].items()],[2.3,1.1,1.1,1.1,1.2])
 fig('spotlight_sample_curves','Figure 2 Official SpotLight platform sample test curves',6.5)
 p('RANGAN reports up to F1=0.83 for network contention on SpotLight [R8]. Our platform sample, features, label prevalence, window length and split differ, so the numbers are contextual rather than a matched win/loss. This experiment establishes that the delivered detector/baseline workflow runs on external measured telemetry; it does not validate SDN recovery or reproduce RANGAN. No 5G-NIDD experiment is claimed.')

 page('6 Confusion matrices and detection interpretation')
 fig('synthetic_confusion','Figure 3 Synthetic test confusion matrix for LightGBM',3.65)
 fig('spotlight_sample_confusion','Figure 4 Official public sample confusion matrix for LightGBM',3.65)
 p('The external matrix exposes the precision cost of detecting rare anomalies. Synthetic separability should not be interpreted as deployment readiness. Normal and benign-burst episode controls additionally report consecutive-alarm event counts and unnecessary changes; sampled-window false positives are kept distinct from false alarms per operational hour.')

 page('7 Primary recovery comparison')
 p('Primary summaries use feasible known-fault families: outage, congestion, loss, delay, concurrent, correlated and cascading. Every load and scenario has equal weight; normal/burst controls, physical disconnections and unseen switch failures are separate. Lower burden and capped unrecovered time are better. Recovery is a service outcome; even B0 can be initially SLA-healthy for a mild injected impairment.')
 for n in (12,20):
  D.add_heading(f'{n} switch topology',2)
  table(['Method','SLA %','Capped s','Recovered %','Median s','p95 s'],[[short(m),percent(lookup[(n,m)]['sla_burden']),fmt(lookup[(n,m)]['capped_unrecovered_seconds'],2),percent(lookup[(n,m)]['restored']),fmt(lookup[(n,m)]['conditional_restoration_median'],2),fmt(lookup[(n,m)]['conditional_restoration_p95'],2)] for m in core],[2.1,1,.95,1.1,.85,.85])
 fig('sla_comparison','Figure 5 Active policy SLA burden with 95% hierarchical confidence intervals',5.8)


 page('8 Fault diversity and recovery distributions')
 p('The full tables include loss, RTT, goodput, collateral burden, route changes and safety events. An advantage on one aggregate does not imply every fault/load stratum benefits; the figures below display those differences explicitly.')
 fig('scenario_comparison','Figure 6 Active-policy scenario burden on each topology; text values expose tied methods. B0 is reported separately.',6.8)
 fig('restoration_cdf','Figure 7 Recovery CDF on the 12-switch known feasible fault set; unrecovered cases remain in the denominator',6.7)
 sw=df[(df.method=='Full_DQN')&(df.scenario=='switch')];inf=df[(df.method=='Full_DQN')&(df.scenario=='infeasible')]
 p(f'Full DQN switch-failure service recovery is {percent(sw[sw.feasible==1].restored.mean())}% among structurally feasible switch cases. Some switch removals disconnect a protected endpoint; these are excluded from feasible-recovery rates and retained in the raw results. Failed-link observations reveal part of the held-out mechanism, so recovery does not demonstrate learned recognition of an unseen class. In the explicit infeasible stress case, {int(inf.restored.sum())} episodes were scored as restored; the no-route condition is not repairable by routing.')

 page('9 Ablation results')
 abl=['Full_DQN','A1_no_diagnosis','A2_no_shield','B2_greedy','A4_predictive','A5_no_topology','A6_no_SHAP','Oracle']
 table(['Variant','SLA %','Capped s','Harm per episode'],[[short(m),percent(lookup[(12,m)]['sla_burden']),fmt(lookup[(12,m)]['capped_unrecovered_seconds'],2),fmt(lookup[(12,m)]['harmful'],4)] for m in abl],[2.8,1.25,1.25,1.6])
 fig('ablations','Figure 8 Matched ablations on the base topology with hierarchical intervals',6.6)
 page('9a Interpretation of component changes')
 for variant,meaning in [('A1_no_diagnosis','diagnosis removal'),('A5_no_topology','topology removal'),('A4_predictive','predictive triggering'),('Oracle','oracle diagnosis')]:
  delta=100*(lookup[(12,variant)]['sla_burden']-full['sla_burden'])
  direction='increased' if delta>0 else 'reduced' if delta<0 else 'did not change'
  p(f'For {meaning}, mean SLA burden {direction} relative to Full (variant minus Full {delta:+.3f} percentage points). This is an observed ablation contrast, not proof that the component has the same effect across all seeds or strata.')
 p('The secondary ablation_statistics.json provides paired hierarchical intervals and Holm-adjusted exploratory contrasts for burden, capped time and harm. This secondary family is kept separate from the planned primary baseline comparisons.')
 p('A1 removes class probabilities and component-risk inputs and retrains. A2 removes the shield and retrains, retaining the residual interlock; its contrast includes the learned-policy response to different constraints. A3 is B2 and is not duplicated as a separate replicate. A5 removes global topology context, neighbour features and candidate-path descriptors, then retrains both the affected supervised models and DQN.')
 p('A4 adds a five-second ridge-style utilization extrapolation with a fixed 0.95 threshold to the frozen full policy; it can respond to the congestion precursor beginning at second 50. The threshold was specified rather than separately optimized; this is a limited predictive extension, not a validated general forecaster. Controls quantify unnecessary actions. A6 uses exactly the full weights with explanation logging off; service results must match exactly, while wall-clock measurements can vary. The oracle gives true labels and component risk to the same full policy; it is privileged diagnostic headroom, not deployable or a separately optimized policy. Sharp oracle scores can be outside the policy training distribution, so this test is not a mathematical performance upper bound.')

 page('10 Safety controls and explanation cost')
 rows=[]
 for m in ('B1_rule','B2_greedy','Full_DQN','A2_no_shield','A4_predictive'):
  a=primary[(primary.topology==12)&(primary.method==m)];installs=(a.route_changes-a.rollbacks).sum()
  rows.append([short(m),int(a.blocked.sum()),int(a.harmful.sum()),fmt(100*a.harmful.sum()/max(1,installs),2),percent(a.collateral_burden.mean())])
 table(['Method','Blocked','Harm','Harm / 100 installs','Collateral %'],rows,[2.05,.9,.7,1.8,1.35])
 fig('safety_speed','Figure 9 Measured harm and restoration speed; compare both axes rather than assuming the shield wins',6.6)
 for m in ('Full_DQN','A6_no_SHAP'):
  a=primary[(primary.topology==12)&(primary.method==m)]
  p(f'{short(m)}: mean episode p95 triggered-decision latency {fmt(a.decision_p95_ms.mean(),3)} ms; mean episode median explanation time {fmt(a.shap_p50_ms.mean(),3)} ms.')
 af=primary[(primary.topology==12)&(primary.method=='Full_DQN')];au=primary[(primary.topology==12)&(primary.method=='A2_no_shield')]
 p(f'The shielded full policy recorded {int(af.harmful.sum())} harm events versus {int(au.harmful.sum())} for the retrained unshielded policy. The burden change on removing the shield was {100*(au.sla_burden.mean()-af.sla_burden.mean()):+.3f} percentage points, and the capped-time change was {au.capped_unrecovered_seconds.mean()-af.capped_unrecovered_seconds.mean():+.3f} seconds. H3 must be assessed from both the harm and service changes; retraining prevents a purely mechanical attribution to the shield alone.')
 p('The explanation implementation uses LightGBM native pred_contrib TreeSHAP, logging the top five signed global-feature contributions after the decision. Its purpose is inspection and overhead measurement. Explanation timing measures contribution computation rather than disk I/O. The decision timing above excludes no-alarm branches and does not include telemetry waiting, physical rule installation or hardware controller work. Shared-host concurrency limits timing comparisons. A one-second decision interval and five-sample verifier preclude millisecond end-to-end restoration claims.')
 controls=df[(df.control==1)&(df.method.isin(['Full_DQN','A4_predictive']))].groupby('method')[['false_alarm_events','unnecessary_changes']].mean()
 p('Control audit averages: '+'; '.join(f'{short(m)} {fmt(v.false_alarm_events,2)} alarm events and {fmt(v.unnecessary_changes,2)} changes per 180-second normal/burst episode' for m,v in controls.iterrows())+'. These controls are not included in the primary recovery test.')

 page('11 DQN learning and seed sensitivity')
 fig('learning_curves','Figure 10 All five validation learning curves for each retrained variant; lower burden is preferable, selection also prioritizes zero harm',6.7)
 table(['Seed','Selected full step','Full SLA %','Full capped s'],[[t['seed'],t['selected_step'],percent(primary[(primary.topology==12)&(primary.method=='Full_DQN')&(primary.training_seed==t['seed'])].sla_burden.mean()),fmt(primary[(primary.topology==12)&(primary.method=='Full_DQN')&(primary.training_seed==t['seed'])].capped_unrecovered_seconds.mean(),2)] for t in validation['training'] if t['variant']=='Full_DQN'],[1.0,2.0,1.8,2.1])
 p('All 20 runs reached 50,000 environment steps. Selected weights can come from an earlier validation checkpoint; final 50,000-step weights are retained separately. Idle monitoring steps are still environment steps, and not every step produces a route decision. This budget is much larger than the original pilot but does not establish convergence or match the Phase 2 500,000-step upper bound. Five validation scenarios plus normal/burst controls guide selection; the final test schedules never update weights or select seeds.')
 p('The per-seed output table reports every retrained variant separately. Environmental repetitions are not counted as independent training replicates. A result driven by one seed should be interpreted through this table, the curves and the hierarchical intervals rather than through a pooled sample-size claim.')

 page('12 Paired statistical analysis')
 p('The primary inferential unit is a reset environmental schedule bundle. Each of 30 bundles averages the seven known-fault scenarios and three loads on the 12-switch topology. The five trained policy seeds rotate evenly, six bundles each. Friedman includes B1, B2, B3, B4 and Full only; B0 is excluded. Paired Wilcoxon contrasts use the same bundles, with Holm correction across both endpoints and all four comparators. Confidence intervals use 10,000 hierarchical resamples of five model-seed groups and environmental bundles within each group.')
 table(['Endpoint','Friedman statistic','p'],[[k,fmt(v['statistic'],3),f'{v["p"]:.3g}'] for k,v in stats['omnibus'].items()],[3.0,2.2,1.7])
 table(['Endpoint','Comparator','Mean delta','95% interval','Holm p'],[[('SLA pp' if t['metric']=='sla_burden' else 'Capped s'),short(t['baseline']),fmt(t['mean_difference']*(100 if t['metric']=='sla_burden' else 1),3),'['+', '.join(fmt(v*(100 if t['metric']=='sla_burden' else 1),3) for v in t['CI95_hierarchical'])+']',f'{t["Holm_p"]:.3g}'] for t in stats['contrasts']],[1.0,1.6,1.0,2.2,1.1])
 p('Delta is Full minus comparator: negative favors Full. The JSON also gives median differences and signed rank-biserial effects. Omnibus rejection alone cannot establish that Full wins; each planned contrast must be read with its sign, uncertainty and adjusted p value. Tests condition on the frozen five trained policies. Only five independent training replicates limit population-level algorithm claims, even when environmental variation is well measured.')
 p('Normal/burst, structurally infeasible, unseen-switch and larger-topology results remain separate from the primary significance family. The design retains 30 blocks as a feasible protocol target; no formal power calculation or post-hoc guarantee is claimed. B3/B4 may tie on this tiny candidate set because both optimizers cover all available actions; this is an action-space limitation rather than extra independent evidence.')

 page('13 Error propagation and representative traces')
 stages=pd.read_csv(R/'stage_transition.csv');g=stages[(stages.topology==12)&(stages.method=='Full_DQN')]
 table(['Cumulative stage','Episodes','Fraction of all','Fraction of prior'],[[a.stage,int(a['count']),fmt(a.fraction_all,3),fmt(a.fraction_previous,3)] for _,a in g.iterrows()],[2.5,1.1,1.6,1.7])
 p('Localisation is the top-one component correctness snapshot at second 65, scored as failure if no trigger occurs at that instant. The cumulative funnel therefore combines an event-level alarm with a fixed-time diagnostic audit; it is not a fully time-aligned causal mediation analysis. Conditional diagnosis and all-fault diagnosis are both retained. The corrected localisation flag compares the predicted component with the withheld affected-component set rather than accepting any nonempty link list.')
 fig('timeline_cascading','Figure 11 One matched high-load cascading episode; the shaded interval is the active fault period',6.6)
 p('The failure CSV retains valid unrecovered runs and assigns the first observed stage from missed detection, localisation snapshot, shield/policy rejection or harm/verification. Several causes can coexist; the classification is an audit heuristic rather than an expert causal diagnosis. Per-scenario/load/topology outputs and additional congestion/switch traces permit inspection of difficult strata. Source truth is joined for evaluation only; it does not enter deployable action selection.')
 p(f'The integrity audit found {audit["shield_blocks_full"]:,} full-policy shield rejections across the complete test grid. B1/B2 have different SLA outcomes in {percent(audit["B1_B2_nonidentical_SLA_fraction"])}% of matched blocks; A6 service equality is {audit["A6_identical_SLA"]}. Thus the former empty shield and identical-baseline issues are explicitly checked, while any remaining ties are reported rather than interpreted as independent gains.')

 if (R/'figures/failure_detail.png').exists():
  page('13a Verification of representative failures')
  fig('failure_detail','Figure 12 Frozen-policy failure replay with RTT, loss, executed actions and shield rejections. Dotted lines are service thresholds.',6.7)
  examples=json.loads((R/'failure_examples_manifest.json').read_text())
  counts={}
  for item in examples:counts[item['category']]=counts.get(item['category'],0)+1
  table(['Observed category','Checked replay examples'],list(counts.items()),[4.5,2.4])
  p('Up to five available cases per observed category were replayed using their original environmental seed and frozen model checkpoint. SLA burden, restoration status, capped time, harm, blocked proposals and route changes were asserted to match the final evaluation. Separate per-second decision and flow traces are retained for each replay. Replays validate deterministic reconstruction of the stochastic schedules; they do not add independent experimental replicates.')

 page('14 Comparison with existing research')
 p('The literature motivates mechanisms rather than interchangeable scores. We implement matched adaptations on the same local recovery environment, and separately test detection on an official public sample. We do not pool synthetic SDN service metrics with RAN intrusion or WSN metrics.')
 table(['Study','Published setting or result','Our executed comparison and fairness limit'],[
 ('R1 Safety-aware RL','5G-NIDD detector accuracy 98.26%; shielded RL [R1].','Full versus retrained A2 measures safety in our routing simulator. Different intrusion task, actions and verifier; no direct accuracy or latency ranking.'),
 ('R3 Threshold DQN','Mininet/ONOS industrial SDN threshold-triggered DQN [R3].','B1 versus Full tests a related threshold/routing mechanism locally. No Mininet or original thermal trace replication; service gains from R3 are not reused as our baseline.'),
 ('R8 RANGAN','SpotLight network-contention F1 up to 0.83 [R8].','Official platform sample LightGBM F1 '+fmt(pub['metrics']['LightGBM']['F1'])+' with matched RF/SVM/IF. Same dataset family, different subset, windows and split; contextual comparison only.'),
 ('R9 LightGBM and FFO','WSN detection and optimizer recovery [R9].','LightGBM detector and GA/PSO-inspired matched local recovery baselines. Different network model and tiny one-flow action space; not an exact GA+RF/PSO+SVM or FFO reproduction.')],[1.2,2.15,3.55])
 p('B3 and B4 use the same objective, candidate action set and 200 ms cap as each other. Their objective is a telemetry proxy combining delay, loss, utilization and change cost; it does not read future outcomes or hidden faults. Because there are only four choices including no-op, both algorithms explicitly cover the candidates before stochastic search. They demonstrate baseline coverage but do not establish large-space optimizer scaling or an advantage of GA over PSO.')
 D.add_heading('Research question interpretation',2)
 h1=all(t['mean_difference']<0 and t['Holm_p']<.05 and t['CI95_hierarchical'][1]<0 for t in stats['contrasts'])
 p('RQ1 and H1: '+('the planned comparisons consistently favor Full on both primary endpoints within the executed simulator.' if h1 else 'the experiment does not establish a consistent Full-policy advantage over all four active baselines on both primary endpoints. The proposed method should not be called uniformly superior.'))
 concurrent=primary[(primary.topology==12)&(primary.scenario=='concurrent')]
 cfull=concurrent[concurrent.method=='Full_DQN'].sla_burden.mean();cno=concurrent[concurrent.method=='A1_no_diagnosis'].sla_burden.mean()
 p(f'RQ2 and H2: concurrent-fault burden was {100*cfull:.3f}% for Full and {100*cno:.3f}% without diagnosis. The descriptive direction '+('favors diagnosis in this subset' if cfull<cno else 'does not favor diagnosis in this subset')+'. This is not a separately corrected confirmatory ablation test. Held-out switch recovery benefits from directly visible link status and is not unknown-class recognition.')
 af=primary[(primary.topology==12)&(primary.method=='Full_DQN')];au=primary[(primary.topology==12)&(primary.method=='A2_no_shield')]
 p(f'RQ3 and H3: Full executed {int(af.harmful.sum())} measured harm events versus {int(au.harmful.sum())} for A2. The observed direction '+('favors the shielded system' if af.harmful.sum()<au.harmful.sum() else 'does not establish fewer harms for the shielded system')+'. Read this together with burden and restoration cost; A2 was retrained, and the simulator harm criterion differs from Phase 2.')
 p('RQ4: component deltas, oracle diagnosis, the stage audit and explanation timing identify where measured performance changes. A6 is intentionally identical in service outcome. These observations quantify contribution within this benchmark; they do not close the gap in real-network, multi-domain validation.')


 page('15 Limitations reproducibility and requirements')
 p('The main limitation is the substituted simulation platform. The service equations omit queues, packet scheduling, real control-plane delays, rule conflicts, emulator saturation and physical instrumentation. Stochastic perturbations improve statistical variation but do not make formula-generated faults realistic. The sample benchmark is one externally measured platform trace; it does not establish public-dataset coverage comparable to RANGAN or R1.')
 p('Preprocessing lacks the planned training-median fallback for long gaps. Detector probability calibration and a validated unknown gate are not implemented. The predictive threshold is fixed, and systematic 5%/10% missingness or observation-delay robustness sweeps remain unexecuted. The 50,000-step DQN budget, small validation grid and tiny action set limit convergence and scalability claims. Decision timings were measured under parallel CPU contention. Hardware CPU/RAM/controller-byte instrumentation, full-controller acknowledgements and a formal power analysis are outside the executed evidence.')
 table(['Teacher component','Evidence in this submission','Qualification'],[
 ('1 Methodology','Pipeline source, trained models, changes in Section 1','Simulation substitution; no specific class feedback.'),('2 Data and preprocessing','Episode/public split manifests; NPZ; counter/missingness trace','Synthetic primary data and one public sample.'),('3 Baselines','Detection baselines; B0-B4; matched outcomes','GA/PSO local adaptations.'),('4 Visualizations','Curves, matrices, seed curves, CDF, safety, ablations','Actual generated outputs only.'),('5 Existing research','Section 14 and external sample experiment','Comparable local adaptations; no exact paper replication.'),('6 Ablations','A1-A6 plus oracle; matched seed budgets where retrained','A3 is B2; A6 deliberately preserves service outputs.'),('7 Discussion','Signed effects, failures, seed sensitivity and limitations','Conclusions restricted to tested simulator and sample.')],[1.5,3.45,1.95])
 p('Run python src/selftest.py and python src/audit.py to check behavior and coverage. Run python src/run.py --analysis-only to regenerate figures and this report from existing measurements, or --full --workers 8 to regenerate the experiments. The README documents dependencies, file meanings and exact commands. Frozen models, selected and final checkpoints, split manifests, raw public inputs, episode outputs and hashes are included. '+str(sum(validation['checks'].values()))+' integrity assertions passed in the delivered run.')
 p('The contribution is an inspectable, reproducible simulation evaluation of a connected detect-diagnose-recover loop with useful controls and ablations. The experiments quantify limits and tradeoffs rather than asserting a predetermined learned-policy advantage. Real SDN emulation and a full external dataset study remain necessary before extending these conclusions beyond the delivered scope.')

 page('References')
 refs=[
 '[R1] W. Al Malwi et al., Safety-Aware Reinforcement Learning for Self-Healing Intrusion Detection in 5G-Enabled IoT Networks, Computers Materials and Continua, 87(2), 2026. doi:10.32604/cmc.2026.074274. https://doi.org/10.32604/cmc.2026.074274',
 '[R3] A. Mwangi et al., A Threshold-Triggered Deep Q-Network-Based Framework for Self-Healing in Autonomic Software-Defined IIoT-Edge Networks, arXiv:2512.14297. https://arxiv.org/abs/2512.14297',
 '[R8] D. Liao, J. Luo, J. Vevstad and N. Pappas, RANGAN GAN-empowered Anomaly Detection in 5G Cloud RAN, 2025. arXiv:2508.20985. https://arxiv.org/abs/2508.20985',
 '[R9] A. Alauthman and A. Al-Hyari, Intelligent Fault Detection and Self-Healing Mechanisms in Wireless Sensor Networks Using Machine Learning and Flying Fox Optimization, Computers, 14(6), 233, 2025. doi:10.3390/computers14060233. https://www.mdpi.com/2073-431X/14/6/233',
 '[S1] C. Sun, U. Pawar, M. Khoja, X. Foukas, M. K. Marina and B. Radunovic, SpotLight Accurate Explainable and Efficient Anomaly Detection for Open RAN, ACM MobiCom 2024. Official dataset repository and platform sample: https://github.com/netsys-edinburgh/SpotLight and https://github.com/netsys-edinburgh/SpotLight/tree/main/sample. Accessed 15 November 2026.',
 '[P2] Provided project methodology phase2_24i0081_24i0005(1).docx. The original R1-R9 identifiers are preserved for continuity.',
 '[P3] Provided teacher brief Phase 03 Implementation.pdf. Its seven required components are mapped in Section 15.'
 ]
 for ref in refs:p(ref)
 D.add_heading('Source and result provenance',2)
 p('Public input SHA256 hashes: platform.csv '+pub['sha256']['platform.csv']+'; label '+pub['sha256']['label']+'. The public experiment manifest records retained row counts, chronological split bounds and all measured metrics. SHA256SUMS.txt covers the final packaged code and artifacts. Results are derived from the accompanying CSV/JSON files; no published paper score is inserted as a locally measured result.')
 for root in (D.element,D.styles.element):
  for border in root.xpath('.//w:pBdr'):border.getparent().remove(border)
 D.styles['Subtitle'].font.italic=False
 for style in D.styles:
  if style.type==1 or style.type==2:
   style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
   style.font.size=Pt(10.5 if style.name in ('Title','Heading 1','Heading 2','Heading 3') else 9)
   if style.type==1:style.paragraph_format.line_spacing=1.5
 for paragraph in D.paragraphs:
  paragraph.paragraph_format.line_spacing=1.0 if paragraph._p.xpath('.//w:drawing') else 1.5
  for run in paragraph.runs:
   run.font.name='Times New Roman';run.font.color.rgb=RGBColor(0,0,0)
   run.font.size=Pt(10.5 if paragraph.style.name in ('Title','Heading 1','Heading 2','Heading 3') else 9)
 for section in D.sections:
  section.different_first_page_header_footer=False
  footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
  footer.paragraph_format.line_spacing=1.5
  run=footer.add_run('Page ');run.font.name='Times New Roman';run.font.size=Pt(9);run.font.color.rgb=RGBColor(0,0,0)
  field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
  run._r.addnext(field)
  footer.add_run(' of ')
  total=OxmlElement('w:fldSimple');total.set(qn('w:instr'),'NUMPAGES');footer._p.append(total)
 for t in D.tables:
  for row in t.rows:
   for cell in row.cells:
    for paragraph in cell.paragraphs:
     paragraph.paragraph_format.line_spacing=1.5
     for run in paragraph.runs:
      run.font.name='Times New Roman';run.font.size=Pt(9);run.font.color.rgb=RGBColor(0,0,0)
 D.save(B/'Phase3_Implementation_Report.docx')
 office=shutil.which('libreoffice') or shutil.which('soffice')
 runtime_office=Path(os.environ.get('CODEX_PRIMARY_RUNTIME_ROOT','/nonexistent'))/'dependencies/bin/override/soffice'
 if office is None and runtime_office.exists():office=str(runtime_office)
 if office:
  subprocess.run([office,'-env:UserInstallation=file:///tmp/cn_phase3_lo_profile','--headless','--convert-to','pdf','--outdir',str(B),str(B/'Phase3_Implementation_Report.docx')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 else:print('DOCX written. Install LibreOffice or export from Word to regenerate the PDF.')
 print('REPORT_WRITTEN',flush=True)
