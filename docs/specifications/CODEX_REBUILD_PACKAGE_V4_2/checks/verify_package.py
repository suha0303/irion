#!/usr/bin/env python3
"""Validate this specification package, not PX4 or an AI implementation.

Run from any directory. Requires the PyYAML version in requirements.txt.
--write-report records document checks and deliberately skips manifest/ZIP checks,
because changing the report changes package hashes. Normal runs are read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable
import zipfile

try:
    import yaml
except ImportError:
    raise SystemExit('Install checks/requirements.txt before running this checker.')

ROOT = Path(__file__).resolve().parents[1]
AI = '02_AI_PREPROCESSING_CODEX_SPEC.md'
SIM = '01_SIMULATION_CODEX_SPEC.md'
SHARED = '00_SHARED_DATA_CONTRACT.md'
REQUIRED = [SHARED, '00_SKILL_LINKS.md', '00_SOURCE_NOTES.md', SIM, AI,
            '03_FINAL_REVIEW.md', '04_CHANGELOG_V3_TO_V4.md', 'AGENTS.md',
            'README.md', '00_SCOPE_AND_REVIEW.md', '04_CHANGELOG_V4_1_TO_V4_2.md', 'checks/verify_package.py', 'checks/requirements.txt',
            'checks/baseline_provenance.json', 'checks/requirements_trace.json']
LINKS = ['https://github.com/alexgreensh/attention-span',
         'https://github.com/multica-ai/andrej-karpathy-skills',
         'https://github.com/dmmulroy/anti-slop',
         'https://github.com/obra/superpowers/blob/main/skills/verification-before-completion/SKILL.md']


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate key: {key}')
        result[key] = value
    return result


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader: UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    return no_duplicates([(loader.construct_object(k, deep=deep),
                           loader.construct_object(v, deep=deep)) for k, v in node.value])


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def finite(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), 'Non-finite serialized number')
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def load_json(text: str) -> Any:
    value = json.loads(text, object_pairs_hook=no_duplicates)
    finite(value)
    return value


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding='utf-8')


def bounded(text: str, start: str, end: str) -> str:
    pos = text.index(start)
    return text[pos:text.index(end, pos + len(start))]


def blocks(text: str) -> list[tuple[str, str]]:
    return re.findall(r'^```(json|yaml)\s*\n(.*?)^```\s*$', text, re.M | re.S)


def run(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def test(name: str, action: Callable[[], Any]) -> None:
        try:
            detail = action()
            checks.append({'name': name, 'status': 'passed', 'detail': detail})
        except Exception as exc:
            checks.append({'name': name, 'status': 'failed', 'detail': f'{type(exc).__name__}: {exc}'})

    test('required_files', lambda: require(all((ROOT/p).is_file() for p in REQUIRED), 'Missing required file'))
    md = {p.name: p.read_text(encoding='utf-8') for p in ROOT.glob('*.md')}
    ai, sim, shared = md[AI], md[SIM], md[SHARED]
    test('utf8_nonempty_no_nul', lambda: require(all(t.strip() and '\x00' not in t for t in md.values()), 'Empty/NUL document'))
    test('markdown_fences_balanced', lambda: require(all(sum(l.startswith('```') for l in t.splitlines()) % 2 == 0 for t in md.values()), 'Unbalanced fences'))
    test('active_document_versions', lambda: require(all('문서 버전: 4.2' in t for n,t in md.items() if n != SHARED), 'Stale document header'))

    def syntax() -> dict[str, int]:
        counts = {'json': 0, 'yaml': 0}
        for name, text in md.items():
            for kind, content in blocks(text):
                try:
                    value = load_json(content) if kind == 'json' else yaml.load(content, Loader=UniqueLoader)
                    finite(value)
                    counts[kind] += 1
                except Exception as exc:
                    raise AssertionError(f'{name}, {kind}: {exc}') from exc
        return counts
    test('json_yaml_examples_syntax_unique_keys_finite', syntax)

    def priority() -> str:
        names = ['00_SKILL_LINKS.md', '00_SOURCE_NOTES.md', AI, SIM, 'README.md', 'AGENTS.md']
        pattern = r'<!-- PRIORITY-SKILLS-BEGIN -->.*?<!-- PRIORITY-SKILLS-END -->'
        values = [re.findall(pattern, md[n], re.S) for n in names]
        require(all(len(v) == 1 for v in values), 'Missing/duplicated priority block')
        require(len({v[0] for v in values}) == 1, 'Priority blocks differ')
        require(all(link in values[0][0] for link in LINKS), 'Exact skill URL missing')
        return '6 identical blocks; four exact original links'
    test('four_skills_six_identical_blocks', priority)

    def contract() -> str:
        for text in (ai, sim):
            matches = re.findall(r'<!-- SHARED-CONTRACT-BEGIN -->\n(.*?)\n<!-- SHARED-CONTRACT-END -->', text, re.S)
            require(len(matches) == 1 and matches[0].rstrip() == shared.rstrip(), 'Shared contract differs')
        return hashlib.sha256(shared.rstrip().encode()).hexdigest()
    test('shared_contract_three_copies', contract)

    def refs() -> dict[str, int]:
        notes = dict(re.findall(r'^\[([SA]\d+)\].*?: (https?://\S+)', md['00_SOURCE_NOTES.md'], re.M))
        result = {}
        expected_ids = {SIM: {f'S{i}' for i in range(1, 12)},
                        AI: {f'A{i}' for i in range(1, 17) if i != 8}}
        for name, expected in expected_ids.items():
            found = re.findall(r'^\[([SA]\d+)\].*?: (https?://\S+)', md[name], re.M)
            require(len(found) == len(expected), f'{name}: incorrect source count')
            require({i for i,_ in found} == expected, f'{name}: source IDs')
            require(all(notes.get(i) == url for i,url in found), f'{name}: source URL mismatch')
            result[name] = len(expected)
        return result
    test('standalone_source_ids_and_urls', refs)

    def local_links() -> int:
        count = 0
        for name, text in md.items():
            for url in re.findall(r'\[[^\]\n]*\]\(([^)\s]+)\)', text):
                if url.startswith(('http:', 'https:', '#')):
                    continue
                path = url.split('#')[0]
                if path in ('MANIFEST.json', 'checks/validation_report.json') and args.write_report:
                    continue
                require((ROOT/path).is_file(), f'{name}: missing local link {path}')
                count += 1
        return count
    test('local_markdown_links_resolve', local_links)

    def trace() -> int:
        data = load_json(read('checks/requirements_trace.json'))
        require(data['count'] == len(data['requirements']) == 68, 'Requirement count mismatch')
        ids = set()
        for item in data['requirements']:
            require(item['id'] not in ids, 'Duplicate requirement')
            ids.add(item['id'])
            lines = read(item['file']).splitlines()
            require(lines[item['line']-1] == item['heading'], f"Changed requirement heading/line {item['id']}")
            require(item['implementation_verified'] is False, 'Document check claims application verification')
            require(item['scope_status'] in ('current_required', 'deferred_not_required'), 'Missing scope status')
            require(all(term in read(item['file']) for term in item.get('required_terms', [])), 'Required scope text missing')
        return len(ids)
    test('requirements_trace_68_positions', trace)

    def source_comparison() -> str:
        data = load_json(read('checks/baseline_provenance.json'))
        require(len(data['standalone_comparisons']) == 2, 'Baseline comparison count')
        require(all(i['standalone_equals_zip'] for i in data['standalone_comparisons']), 'Mismatch in baseline evidence')
        require(re.fullmatch('[a-f0-9]{64}',data['baseline_zip_sha256']) is not None, 'Invalid baseline hash')
        return 'Recorded two direct v4.1 standalone-vs-ZIP byte comparisons'
    test('recorded_baseline_provenance', source_comparison)

    def unchanged() -> int:
        data = load_json(read('checks/baseline_provenance.json'))
        for item in data['preserved_sections']:
            value = bounded(read(item['file']), item['start'], item['end_prefix']).encode()
            require(len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256'], 'Changed preserved section: '+item['start'])
        return len(data['preserved_sections'])
    test('seven_preserved_technical_sections', unchanged)
    test('input_profile_dimensions', lambda: require(sum([4,3,3,2,3,1,1,1,1]) == 19 and 19-3-1 == 15 and all(x in ai for x in ['common19_ctx6','common15_ctx6','common19_ctx0','common15_ctx0']), 'Input feature dimensions/profile'))

    parsed_json = [load_json(content) for kind, content in blocks(ai) if kind == 'json']
    cfg = next(yaml.load(content, Loader=UniqueLoader) for kind,content in blocks(ai) if kind == 'yaml' and 'experiment: common19_forecast_reference_v4' in content)
    fc = next(v for v in parsed_json if v.get('schema') == 'drone.forecast/3.0')
    dc = next(v for v in parsed_json if v.get('schema') == 'drone.prediction/3.0')

    def config() -> None:
        require((cfg['task'],cfg['sampling_hz'],cfg['window_samples'],cfg['horizon_samples'],cfg['context_seconds'],cfg['infer_every_seconds']) == ('forecast',10,100,10,60,1), 'Default config mismatch')
        require(cfg['target_names'] == ['body_rate_p','body_rate_q','body_rate_r','pack_voltage'], 'Targets mismatch')
        require(cfg['training']['noise_std_scaled'] == 0, 'Unexpected input noise in forecast baseline')
        require(sum(cfg['loss_weights'].values()) == 1, 'Group weights do not sum to one')
        require(not cfg['forecast_policy']['future_observed_inputs'] and not cfg['forecast_policy']['teacher_forcing'], 'Future forward input enabled')
    test('forecast_default_config', config)
    test('forecast_shape_example', lambda: require(len(fc['target_times_us']) == len(fc['predicted_values']) == 10 and all(len(row) == 4 for row in fc['predicted_values']) and fc['target_names'] == cfg['target_names'], 'Forecast [H,4] shape'))

    def timeline() -> dict[str, Any]:
        t = fc['origin_time_us']; step = 100000
        history = [t-step*k for k in range(99,-1,-1)]
        targets = [t+step*k for k in range(1,11)]
        require(len(history) == 100 and history[-1]-history[0] == 9900000, 'History interval')
        require(not(set(history) & set(targets)) and targets == fc['target_times_us'], 'History overlaps targets')
        require(t <= fc['issued_time_us'] < targets[0], 'Late example forecast')
        require(fc['last_input_sample_time_us'] <= fc['input_available_cutoff_us'] == t, 'Input cutoff example')
        require(dc['forecast_id'] == fc['forecast_id'], 'Forecast-decision join mismatch')
        require(dc['target_start_time_us'] == targets[0] and dc['target_end_time_us'] == targets[-1], 'Decision target span')
        require(dc['decision_time_us'] >= dc['last_target_available_time_us'] >= targets[-1], 'Score backdated')
        return {'history_extent_seconds':9.9, 'forecast_horizon_seconds':1.0, 'targets':10, 'scope':'document example arithmetic only'}
    test('forecast_and_decision_time_arithmetic', timeline)

    def reference_arithmetic() -> str:
        # Reference predicates, deliberately not imports from an application.
        t, targets = 0, [100000*k for k in range(1,11)]
        require(not all(ts <= 900000 for ts in targets), 'Reference block scored before all targets')
        require(all(ts <= 1000000 for ts in targets), 'Reference block not mature at final target')
        require(not (100001 < targets[0]), 'Reference late policy')
        require((61000000-60000000)/1e6 == 1, 'Horizon delay arithmetic')
        actual = [20.00-0.01*k for k in range(10)]
        forecast = actual.copy()
        mse = sum((x-y)**2 for x,y in zip(actual,forecast))/len(actual)
        require(mse == 0, 'Illustrative trend counterexample arithmetic')
        return 'Reference time predicates and zero-error example only; no learned-model or physical diagnostic claim'
    test('reference_only_maturity_late_and_error_examples', reference_arithmetic)
    test('unestablished_health_in_output_example', lambda: require(dc['diagnosis_status'] == 'not_established' and dc['health_estimate'] == {'soh':None,'rul':None}, 'Health claim inferred from score'))

    def onset() -> str:
        vals = [load_json(v) for k,v in blocks(shared) if k == 'json']
        events = [v for v in vals if v.get('schema') == 'drone.evaluation.event/2.1']
        pre = next(v for v in events if v['onset_kind'] == 'pre_existing')
        acute = next(v for v in events if v['onset_kind'] == 'in_flight_step')
        require(pre['physical_onset_time_us'] is None and pre['applied_time_us'] == 0, 'Pre-existing onset misrepresented')
        require(acute['requested_time_us'] <= acute['applied_time_us'] == acute['physical_onset_time_us'], 'Acute application chronology')
        return 'Two event forms explicitly separated'
    test('existing_and_acute_event_examples', onset)
    test('raw_evaluation_output_schema_boundaries', lambda: require('drone.raw.event/2.0' in shared and 'drone.raw.watermark/2.0' in shared and 'drone.evaluation.event/2.1' in shared and 'drone.prediction/2.0' not in ai, 'Mixed schema semantics'))
    test('sensor_exclusions_in_both_specs', lambda: require(all(word in text for text in (ai,sim) for word in ['개별 RPM','온도','개별 셀 전압','BAT.Res','truth']), 'Sensor exclusion missing'))
    test('normal_health_roles', lambda: require(all(term in ai and term in sim for term in ['confirmed_healthy','uncertain','abnormal']) and all(term in ai for term in ['정상 train','정상 validation','정상 calibration','독립 정상 test','독립 이상 test']), 'Normal-only roles absent'))
    test('split_and_longitudinal_protocols', lambda: require(all(term in ai and term in shared for term in ['same_asset_future','paired_case_id']) and 'x500_run_holdout' in ai and 'asset_id' in ai and '교체' in ai, 'Protocol/asset boundaries missing'))
    test('context_and_filter_memory_limits', lambda: require(all(term in ai for term in ['60초 통계','segment 누적량','필터의 내부 상태','9.9초','ctx0']), 'Memory range not stated'))
    def forecast_only_scope() -> str:
        require('## 18.' not in ai and '18절' not in ai, 'Deleted section or reference remains')
        for name in [AI, SIM, SHARED, 'README.md', 'AGENTS.md', '00_SOURCE_NOTES.md']:
            require(re.search(r'오토인코더|autoencoder|reconstruct|LSTM-AE|Transformer-AE|\bPCA\b|(?<![\w.])AE(?![\w.])|재구성', md[name], re.I) is None,
                    f'{name}: removed comparison task remains')
        require('## 8. LSTM 미래 예측기' in ai and '## 9. Transformer 미래 예측기' in ai,
                'Forecast model specifications missing')
        return 'Section 18 and related comparison instructions removed; forecast models retained'
    test('forecast_scope_and_removed_comparison_references', forecast_only_scope)
    test('forecast_score_targets_only', lambda: require(all(term in bounded(ai,'### 11.1','### 11.2') for term in ['reference_error_scale[j,h]','[p,q,r]','[pack_voltage]','예측하지 않는 전류']), 'Score/target mismatch'))
    test('immutable_pending_and_no_backdating', lambda: require(all(term in ai for term in ['PendingForecastStore','불변 저장','forecast_late','score를 forecast origin으로 소급 기록하지 않는다','중복 발행되지']), 'Time/state requirements missing'))
    test('normalization_and_adaptation_guard', lambda: require(all(term in ai for term in ['정상 train split','무감독으로 계속 적응시키는 기능은 기본값에서 끈다','비행별/윈도별']), 'Healthy baseline protection missing'))
    test('normal_abnormal_test_matrix', lambda: require(all(f'| {name} ' in ai for name in ['N1','N2','F1','F2','F3','F4','F5','O1','Q1']), 'Missing test family'))
    test('simulator_preexisting_and_severity_boundaries', lambda: require(all(term in sim for term in ['pre_existing','in_flight_step','in_flight_ramp','같은 초기 SOC와 같은 초기 잔여 Ah','severity sweep','시간 함수']), 'Physical evaluation limitation missing'))
    test('conditional_diagnostic_is_separate_task', lambda: require('conditional_response' in ai and '같은 시점 전류를 사용했다면 미래를 미리 맞혔다고 보고하지 않는다' in ai, 'Conditional diagnosis presented as forecast'))
    test('forecast_cli_and_baselines', lambda: require('--model lstm --task forecast' in ai and '--model transformer --task forecast' in ai and 'capstone-ai trend' in ai and 'capstone-ai baseline --model persistence' in ai and '--model ridge_forecast --task forecast' in ai, 'CLI/task mismatch'))
    test('implementation_scope_not_claimed_complete', lambda: require('이번 작업에서 실행하지 않았다' in md['03_FINAL_REVIEW.md'] and '실제 모델 실행 결과가 아니다' in ai and '원본 v3는 그대로 남겼다' in md['03_FINAL_REVIEW.md'], 'Scope boundary not stated'))
    test('no_unresolved_authoring_placeholders', lambda: require(all('__VALIDATION_SUMMARY__' not in t and '\ufffd' not in t for t in md.values()), 'Unresolved authoring marker/decoding error'))
    test('px4_and_ai_preprocessing_responsibility', lambda: require('기술적으로 유지할 대상은 PX4' in sim and '**전처리 전체는 이 AI 파트가 구현한다.**' in ai and '안전 제어를 우회하지 않는다' in shared, 'Responsibility/safety boundary missing'))

    def current_scope() -> str:
        names = [AI, SIM, 'README.md', 'AGENTS.md', '00_SCOPE_AND_REVIEW.md']
        values = [re.findall(r'<!-- CURRENT-SCOPE-BEGIN -->.*?<!-- CURRENT-SCOPE-END -->', md[n], re.S) for n in names]
        require(all(len(v) == 1 for v in values) and len({v[0] for v in values}) == 1, 'Current scope blocks differ')
        require(all(term in values[0][0] for term in ['X500 기준 기체 한 구성', '장기 목표', '현재 필수 구현 및 완료 조건에서 제외', 'not_in_scope']), 'Scope boundaries missing')
        return 'Five identical current-scope blocks: X500 now, foundation/customization later'
    test('x500_current_scope_five_copies', current_scope)
    test('x500_shared_contract_scope', lambda: require('## C0. 현재 적용 범위' in shared and all(x in shared for x in ['x500_run_holdout', '현재 인수 조건이 아니다', 'not_in_scope']), 'Shared scope missing'))
    test('current_run_not_unseen_vehicle_required', lambda: require('| `x500_run_holdout` 현재 기본' in ai and '`unseen_asset`이나 기종 전체 holdout은 후속 선택 평가' in ai and cfg['split_protocol']=='x500_run_holdout', 'Current default requires a new vehicle'))
    test('single_vehicle_and_pack_scope', lambda: require(cfg['vehicle_profile']=='x500_reference' and '기준 팩 하나를 선택' in sim and '모터 4개' in ai, 'Current vehicle/profile scope expanded'))
    def current_cli() -> None:
        cli = bounded(ai, '## 15. 실제 제공할 CLI', '## 16. 자동 시험과 합격 기준')
        require('flight.BIN' not in cli and 'configs/sources/company.yaml' not in cli and 'fine-tune' not in cli, 'Future adaptation path in current CLI')
        require('native_logs/flight.ulg' in cli and 'configs/sources/px4.yaml' in cli, 'X500 CLI missing')
        require('회사 이전 도구`' not in ai, 'Company transfer remains in required milestones')
    test('x500_cli_and_milestones', current_cli)
    test('foundation_roadmap_not_current_deliverable', lambda: require('## 14. 장기 목표와 후속 확장 — 현재 범위 제외' in ai and '범용 fine-tuning API는 추가하지 않는다' in ai and '회사 기체·다기체 사전학습·기종 간 이전·실기체 검증은 이번 필수 합격 조건이 아니다' in sim, 'Deferred work remains mandatory'))
    test('fallback_profiles_not_all_required', lambda: require('두 profile을 모두 학습·배포하는 것이 필수는 아니다' in ai and '모든 대체/비교 profile을 필수로 학습·배포하지 않는다' in shared, 'Unnecessary mandatory profiles'))
    test('history_not_active_instruction', lambda: require('현재 구현 지시가 아니다' in md['04_CHANGELOG_V3_TO_V4.md'] and '과거 이력만 보관' in md['README.md'], 'Historical requirements can override current scope'))

    if not args.write_report:
        def manifest() -> int:
            data = load_json(read('MANIFEST.json'))
            require(data['version'] == '4.2', 'Manifest version')
            actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'MANIFEST.json' and '__pycache__' not in p.parts}
            listed = {v['path'] for v in data['files']}
            require(actual == listed and len(listed) == len(data['files']), 'Manifest inventory mismatch')
            for item in data['files']:
                p = ROOT/item['path']; raw = p.read_bytes()
                require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'Manifest hash mismatch: '+item['path'])
            return len(listed)
        test('manifest_all_file_hashes', manifest)
    if args.zip:
        def archive() -> int:
            with zipfile.ZipFile(args.zip) as z:
                require(z.testzip() is None, 'ZIP CRC error')
                entries = [n for n in z.namelist() if not n.endswith('/')]
                require(len(entries) == len(set(entries)), 'Duplicate ZIP paths')
                expected = {ROOT.name+'/'+p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
                require(set(entries) == set(expected), 'ZIP file list mismatch')
                for name,path in expected.items():
                    require(z.read(name) == path.read_bytes(), 'ZIP bytes differ: '+name)
                return len(entries)
        test('zip_crc_inventory_and_bytes', archive)

    report = {'schema':'codex.document.validation/1.0','package_version':'4.2',
              'executed_at_utc':datetime.now(timezone.utc).isoformat(),
              'command': 'python checks/verify_package.py' + (' --write-report' if args.write_report else '') + (' --zip '+str(args.zip) if args.zip else ''),
              'scope':'specification files, serialized examples, reference arithmetic, requirements mapping, and optional archive hashes only',
              'application_tests_executed':False,'python_version':sys.version.split()[0],'pyyaml_version':yaml.__version__,
              'passed':sum(c['status']=='passed' for c in checks),'failed':sum(c['status']=='failed' for c in checks),
              'checks':checks}
    if args.write_report:
        (ROOT/'checks/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['failed'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report',action='store_true',help='Write document-only report; run before final hashing')
    parser.add_argument('--zip',type=Path,help='Validate the complete delivered ZIP against this directory')
    args = parser.parse_args()
    if args.write_report and args.zip:
        parser.error('--write-report and --zip cannot be combined')
    raise SystemExit(run(args))
