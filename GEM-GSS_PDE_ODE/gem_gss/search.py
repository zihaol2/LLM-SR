"""Bounded frontier -> search/history/repair -> bounded frontier; one final choice."""
import random
import time
from . import core as C, hypothesis, rotation, feedback
from .api import once
from .compression import make_prompt
from .config import field_budget
from .numerics import fit, final_equation, make_data, restore_models, statistics as numerical_statistics
from .graph import Registry
from .memory import Archive, fit_version
from .probes import Probes
from .refinement import FullRefiner
from .frontier import update
from .final_selection import cards_for, validate
from .wire import expand_response_expressions


def execute(archive, proposal, number, progress=None):
    model,row = feedback.evaluate(archive,proposal)
    row['state'] = archive.record(model,proposal,number)
    row['model'] = model
    if progress is not None and archive.data.fit_calls % 25 == 0:
        progress()
    return row


def public(row):
    return {k:v for k,v in row.items() if k!='model'}


def run(case,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'final_result.json').exists():
        return C.read(out/'final_result.json')
    cp=C.read(out/'checkpoint.json') if (out/'checkpoint.json').exists() else None
    started=cp['started_unix'] if cp else time.time()
    data=make_data(case); registry=Registry(case,cp['graph'] if cp else None)
    archives={field:Archive(data,field,registry) for field in case['fields']}
    rng=random.Random(C.SETTINGS['seed']); probes=Probes()
    groups=cp['rotation_groups'] if cp else rotation.partition(registry.graph.candidates,rng)
    history,stats,pending,plan,selected=[],{},None,None,None
    if cp:
        if cp.get('schema')!='bounded-frontier-v1':
            raise ValueError('71 requires a fresh run, not a 70 checkpoint')
        history,stats,pending,plan=cp['history'],cp['stats'],cp['pending'],cp['next_plan']
        selected=cp.get('final_selection')
        for field,archive in archives.items(): archive.restore(cp['archives'][field])
        def tuples(v): return tuple(tuples(i) for i in v) if isinstance(v,list) else v
        rng.setstate(tuples(cp['rng_state']))
        for k,v in cp['counters'].items(): setattr(data,k,v)
        restore_models(data,archives,(cp.get('numerical_backend_stats') or {}).get('counts'))
    else:
        for field,archive in archives.items():
            archive.add(fit(data,field,[],registry),None,dict(operation='root'),0)
        C.dump(out/'initial_graph.json',registry.export())
    phase=dict(stage='initializing',round=len(history)+1,field=None)
    def progress(**updates):
        phase.update(updates)
        C.dump(out/'progress.json',dict(**phase,completed_rounds=len(history),fit_calls=data.fit_calls,
            working_set_sizes={f:len(a.active_ids) for f,a in archives.items()},updated_unix=time.time(),
            numerical_backend_stats=numerical_statistics(data)))
    def save():
        C.dump(out/'checkpoint.json',dict(schema='bounded-frontier-v1',started_unix=started,
            graph=registry.export(),history=history,stats=stats,pending=pending,next_plan=plan,
            final_selection=selected,rotation_groups=groups,rng_state=rng.getstate(),
            counters={k:getattr(data,k) for k in ['fit_calls','fit_nfev','feature_calls']},
            archives={f:a.export() for f,a in archives.items()},numerical_backend_stats=numerical_statistics(data)))
    def prompt_after(number,event=None):
        sampled=rotation.group_for(groups,number+1) if number<C.SETTINGS['rounds'] else []
        preview={}
        if sampled:
            for field,archive in archives.items():
                queues,info=hypothesis.history_candidates(archive,sampled,[],probes)
                preview[field]=dict(history_mechanisms=info['history_mechanisms'],
                    examples=[dict(support=list(item['support']),relative_probe_MSE=item['probe_relative_MSE'])
                              for queue in queues for item in list(queue)[:2]])
        text,packet=make_prompt(case,registry,archives,sampled,stats,history+([event] if event else []),number,preview)
        C.dump(out/'inputs'/f'{number:02d}_graph.json',packet)
        return text
    def working_snapshot(number,audits=None):
        C.dump(out/'working_sets'/f'{number:02d}.json',dict(round=number,
            fields={field:[dict(state=s['id'],model_version=s['model_version'],model=s['model'])
                           for s in archive.view()] for field,archive in archives.items()},audits=audits))
    if plan is None and not history:
        if pending is None:
            working_snapshot(0)
            pending=dict(phase='bootstrap',round=0,prompt=prompt_after(0));save()
        if pending['phase']!='bootstrap': raise RuntimeError('Invalid bootstrap checkpoint')
        progress(stage='bootstrap_planning',round=0)
        raw,record=once(pending['prompt'],out/'api',0)
        plan=raw if record['status']=='accepted' else {}
        pending=None;save()
    for number in range(len(history)+1,C.SETTINGS['rounds']+1):
        if pending is None:
            begin=time.time();sampled=rotation.group_for(groups,number)
            input_ids={f:list(a.active_ids) for f,a in archives.items()}
            for mid in sampled: stats.setdefault(mid,dict(sampled=0,tested=0))['sampled']+=1
            sent_packet=C.read(out/'inputs'/f'{number-1:02d}_graph.json')
            expanded_plan,template_audit=expand_response_expressions(plan,sent_packet.get('wire_packet',{}))
            policy=hypothesis.normalize_policy(expanded_plan,archives)
            new=hypothesis.register_mechanisms(policy,registry,number)
            coverage,historical,revisions,edit_audits,selections,frontier_audits={},{},{},{},{},{}
            for field,archive in archives.items():
                current_ids=[]
                def record_trial(proposal, category):
                    row=execute(archive,proposal,number,progress)
                    category.append(public(row))
                    if row['state'] is not None: current_ids.append(row['state'])
                    return row
                progress(stage='coverage',round=number,field=field)
                proposals,selections[field]=hypothesis.build_tests(policy,archive,sampled,rng,number)
                coverage[field]=[]
                for proposal in proposals:
                    if proposal['parent'] not in input_ids[field]: raise RuntimeError('Undisplayed coverage parent')
                    proposal['kind']='coverage_'+proposal['source']
                    record_trial(proposal,coverage[field])
                tested={mid for row in coverage[field] for mid in row['support']} & set(sampled)
                if not set(selections[field]['selected'])<=tested:
                    raise RuntimeError('Coverage quota not executed')
                selections[field]['actually_tested_sampled']=sorted(tested)
                revisions[field]=[];edit_audits[field]=[]
                def evaluate(proposal): return record_trial(proposal,revisions[field])
                progress(stage='LLM_skeleton_edits')
                explicit=[]
                for proposal in hypothesis.llm_revisions(archive,policy,new,audit=edit_audits[field],number=number):
                    row=evaluate(proposal);explicit.append(row)
                    edit_audits[field][proposal['edit_audit_index']].update(
                        status='fitted' if row['valid'] else 'fit_invalid',state=row['state'],score=row['score'])
                progress(stage='history_recombination')
                queues,_=hypothesis.history_candidates(archive,sampled,new['ids'],probes)
                stream=hypothesis.next_unseen(queues,archive);historical[field]=[]
                for _ in range(field_budget(C.SETTINGS['history_fits_per_round'],case['fields'],field)):
                    proposal=next(stream,None)
                    if proposal is None: break
                    proposal['kind']='historical_'+proposal['origin']
                    record_trial(proposal,historical[field])
                progress(stage='full_refinement')
                refiner=FullRefiner(archive,number,evaluate)
                for row in explicit:
                    if row['valid']: refiner.reduce(row)
                preliminary=update(archive,input_ids[field],current_ids,C.SETTINGS['frontier_limit'])
                family=refiner.run()
                C.dump(out/'refinement'/f'round{number:02d}_{field}.json',dict(preliminary=preliminary,**family))
                frontier_audits[field]=update(archive,input_ids[field],current_ids,C.SETTINGS['frontier_limit'])
            rows=[row for group in [coverage,historical,revisions] for batch in group.values() for row in batch]
            for mid in {mid for row in rows for mid in row['support']}:
                stats.setdefault(mid,dict(sampled=0,tested=0))['tested']+=1
            event=dict(round=number,input_working_ids=input_ids,
                output_working_ids={f:list(a.active_ids) for f,a in archives.items()},
                frontier_audits=frontier_audits,coverage_fits=sum(map(len,coverage.values())),
                history_fits=sum(map(len,historical.values())),revision_fits=sum(map(len,revisions.values())),
                valid_fits=sum(row['valid'] for row in rows),ordinary_local_fits=0,
                edit_counts={name:sum(item['status']==status for batch in edit_audits.values() for item in batch)
                             for name,status in [('fitted','fitted'),('rejected','rejected'),('fit_invalid','fit_invalid')]},
                policy_normalization_errors=policy['errors'],numerical_seconds=time.time()-begin)
            event['template_response_expansion']=template_audit
            event['edit_counts']['proposed']=sum(map(len,edit_audits.values()))
            if event['history_fits']>C.SETTINGS['history_fits_per_round']: raise RuntimeError('History budget exceeded')
            C.dump(out/'rounds'/f'{number:02d}.json',dict(event=event,selection=selections,
                coverage=coverage,historical=historical,revisions=revisions,LLM_edit_audit=edit_audits))
            working_snapshot(number,frontier_audits)
            final=number==C.SETTINGS['rounds']
            pending=dict(phase='final_selection' if final else 'planning',round=number,event=event,
                cards={f:cards_for(a) for f,a in archives.items()} if final else None,prompt=prompt_after(number,event))
            save()  # Numeric round + exact working set frozen before any request.
        if pending['round']!=number: raise RuntimeError('Checkpoint round mismatch')
        progress(stage=pending['phase'],round=number,field=None)
        raw,record=once(pending['prompt'],out/'api',number)
        if pending['phase']=='final_selection':
            if record['status']!='accepted':
                progress(stage='final_selection_unavailable')
                raise RuntimeError('Final LLM unavailable; working set saved; no implicit Score fallback or retry')
            selected=validate(raw,pending['cards'])
            for field,decision in selected.items():
                archive=archives[field]
                if decision['state'] not in archive.active_ids or fit_version(archive.states[decision['state']]['model'])!=decision['model_version']:
                    raise RuntimeError('Final choice is not a frozen working-set fit')
            C.dump(out/'final_selection.json',dict(policy=raw,decisions=selected,fit_calls=data.fit_calls,
                reference_access=False,final_fits=0))
        event=pending['event'];event.update(API_status=record['status'],reason=raw.get('reason',''))
        history.append(event);plan=raw if record['status']=='accepted' else {};pending=None;save()
        print(case['id'],'ROUND COMPLETE',number,'working set',{f:len(a.active_ids) for f,a in archives.items()},
              'coverage/history/revision',event['coverage_fits'],event['history_fits'],event['revision_fits'],flush=True)
    if selected is None: raise RuntimeError('Missing explicit final selection')
    roots={mid for group in groups for mid in group}
    coverage={f:len(roots & a.tested_ids())/len(roots) if roots else 1. for f,a in archives.items()}
    if C.SETTINGS.get('coverage_schedule')=='global_1pct':
        if C.SETTINGS['rounds']==20 and any(v<.2 for v in coverage.values()):
            raise RuntimeError('Reduced-budget exploration did not reach 20% initial-root coverage')
    elif C.SETTINGS['rounds']==20 and any(v!=1. for v in coverage.values()): raise RuntimeError('Incomplete initial-root coverage')
    records=[C.read(path) for path in sorted((out/'api').glob('*_record.json'))]
    before=data.fit_calls
    equations={f:final_equation(data,f,archives[f].states[d['state']]['model']) for f,d in selected.items()}
    result=dict(method='GEM-GSS bounded frontier iteration',completed_rounds=len(history),anonymous_id=case['id'],
        equations=equations,selected_states={f:d['state'] for f,d in selected.items()},
        final_working_ids={f:list(a.active_ids) for f,a in archives.items()},final_selection=selected,
        final_selection_rule='One LLM selects an existing final-frontier equation by MSE/complexity marginal tradeoff',
        final_API_calls=records[-1]['http_attempts'],search_API_calls=sum(r['http_attempts'] for r in records[:-1]),
        total_API_calls=sum(r['http_attempts'] for r in records),bootstrap_API_calls=records[0]['http_attempts'],
        round_planning_API_calls=sum(r['http_attempts'] for r in records[1:-1]),terminal_fit_calls=0,
        unavailable_LLM_records=[r['round'] for r in records if r['status']!='accepted'],
        fit_calls=data.fit_calls,loss_evaluations=data.fit_nfev,
        fit_breakdown=dict(initial_empty=len(archives),coverage=sum(r['coverage_fits'] for r in history),
            history=sum(r['history_fits'] for r in history),online_revision=sum(r['revision_fits'] for r in history),ordinary_local=0,terminal=0),
        initial_root_attempt_coverage=coverage,coverage_schedule=C.SETTINGS.get('coverage_schedule','legacy_half'),
        guaranteed_initial_root_fraction=.2 if C.SETTINGS.get('coverage_schedule')=='global_1pct' else 1.,
        MSE=sum(e['MSE'] for e in equations.values())/len(equations),
        reference_access=False,frozen_unix=time.time(),seconds=time.time()-started,
        numerical_backend_stats=numerical_statistics(data))
    if data.fit_calls!=before or sum(result['fit_breakdown'].values())!=data.fit_calls: raise RuntimeError('Fit accounting mismatch')
    C.dump(out/'final_graph.json',registry.export());C.dump(out/'final_result.json',result)
    progress(stage='complete',round=len(history))
    return result
