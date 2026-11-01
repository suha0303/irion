import { Activity, ArrowUpRight, Check, CircleAlert, Download, FileCheck2, FileUp, FlaskConical, Info, Plug, ShieldCheck } from 'lucide-react';
import type { Detection, Frame, Manifest, Metrics, Point } from './types';
import { detectionKey, numeric, statusLabel, targetLabel, timeLabel } from './types';
import { TimeChart, motorColors } from './Charts';

export function EventList({ events, selected, choose }: { events: Detection[]; selected: string | null; choose: (e: Detection) => void }) {
  if (!events.length) return <div className="empty-state"><ShieldCheck size={30}/><strong>아직 표시할 판정이 없습니다</strong><p>AI 미연결은 정상 판정이 아닙니다.<br/>결과는 판정 시각에 도달하면 나타납니다.</p></div>;
  return <div className="event-list">{[...events].reverse().map(e => <button className={`event-row ${selected === detectionKey(e) ? 'selected' : ''}`} key={detectionKey(e)} onClick={() => choose(e)}>
    <span className={`event-icon ${e.status === 'normal' ? 'green' : 'amber'}`}>{e.status === 'normal' ? <Check size={17}/> : <CircleAlert size={17}/>}</span>
    <span className="event-copy"><strong>{targetLabel[e.target]??e.target} · {statusLabel[e.status]}</strong><small>{timeLabel(e.decided_at_ms, true)} <span>•</span> {e.provider_id ?? 'legacy'} · {e.origin === 'fixture' ? '시험용 응답' : e.model_version}</small></span><ArrowUpRight size={16}/>
  </button>)}</div>;
}

export function EventDetail({ event, points, cursor }: { event: Detection | null; points: Point[]; cursor: number }) {
  if (!event) return <section className="panel event-detail"><div className="empty-state tall"><Activity size={36}/><strong>판단의 근거를 함께 확인하세요</strong><p>왼쪽 이벤트를 선택하면 발생 구간과<br/>관련 관측값을 같은 시간축에서 볼 수 있습니다.</p></div></section>;
  const battery = event.target === 'battery';
  return <section className="panel event-detail">
    <div className="panel-heading"><div><span className="eyebrow">DETECTION EVIDENCE</span><h2>{targetLabel[event.target]??event.target} · {statusLabel[event.status]}</h2></div><span className={`pill ${event.origin === 'fixture' ? 'amber' : 'teal'}`}>{event.origin === 'fixture' ? '시험용 응답' : '모델 결과'}</span></div>
    <p className="detail-description">{event.description || event.title}</p>
    {event.kind !== 'explanation' && <div className="detail-stats"><div><small>이상 점수</small><strong>{numeric(event.score, 3)}</strong></div><div><small>판정 임계값</small><strong>{numeric(event.threshold, 3)}</strong></div><div><small>근거 구간</small><strong className="small-value">{timeLabel(event.start_ms, true)} — {timeLabel(event.end_ms, true)}</strong></div></div>}
    <div className="chart-caption"><span>{battery ? '배터리 전류' : '모터 RPM'} · 관측값</span><span className="muted">음영: 모델이 제시한 근거 구간</span></div>
    <TimeChart points={points} cursor={cursor} event={event} signals={battery ? [{ key: 'battery_current_a', name: '팩 전류' }] : [1, 2, 3, 4].map(i => ({ key: `motor_${i}_rpm`, name: `모터 ${i}` }))} colors={battery ? ['#0ca897'] : motorColors} unit={battery ? 'A' : 'rpm'} height={240}/>
    {event.evidence.length ? <div className="table-scroll"><table><thead><tr><th>근거 신호</th><th>관측값</th><th>예측값</th><th>잔차</th><th>설명</th></tr></thead><tbody>{event.evidence.map((e, i) => <tr key={i}><td>{e.signal}</td><td>{numeric(e.observed)} {e.unit}</td><td>{numeric(e.predicted)} {e.unit}</td><td>{numeric(e.residual)} {e.unit}</td><td>{e.note || '—'}</td></tr>)}</tbody></table></div> : <div className="note"><Info size={16}/>예측값과 잔차가 제공되지 않았습니다. 위 차트는 관측값입니다.</div>}
    <dl className="detail-meta"><div><dt>모델 버전</dt><dd>{event.model_version}</dd></div><div><dt>판정 시각 / 유효 종료</dt><dd>{timeLabel(event.decided_at_ms, true)} / {timeLabel(event.valid_until_ms, true)}</dd></div><div><dt>Epoch</dt><dd>{event.epoch}</dd></div><div><dt>입력 SHA-256</dt><dd className="hash">{event.input_ref?.sha256 ?? event.input_ref?.segment_sha256 ?? event.input_sha256 ?? '—'}</dd></div></dl>
  </section>;
}

export function Diagnostics({run,frame,metrics,connectAI}:{run:Manifest;frame:Frame|null;metrics:Metrics|null;connectAI:()=>void}) {
  const valid=frame?.streams.filter(s=>s.status==='valid').length??0;
  const period=frame?.input_contract?.sample_period_ms??run.profile?.sample_period_ms;
  return <div className="diagnostics"><div className="diagnostic-kpis">
    <div className="panel diagnostic-kpi"><FileCheck2 size={21}/><small>수집·재생 관측</small><strong>{run.accepted_records.toLocaleString()}<span> / {run.total_records.toLocaleString()}</span></strong></div>
    <div className="panel diagnostic-kpi"><CircleAlert size={21}/><small>검사에서 제외</small><strong>{run.excluded_records}<span>건</span></strong></div>
    <div className="panel diagnostic-kpi"><Activity size={21}/><small>현재 유효 스트림</small><strong>{valid}<span> / {frame?.streams.length??run.streams.length}종</span></strong></div>
    <div className="panel diagnostic-kpi"><Activity size={21}/><small>전처리 출력 주기</small><strong>{period?numeric(1000/period,1):'—'}<span>{period?'Hz':'원본 입력'}</span></strong></div>
  </div>
  <section className="panel"><div className="panel-heading"><div><span className="eyebrow">STREAM QUALITY</span><h2>수신 상태와 주기</h2></div><span className="muted">재생 시각 {timeLabel(frame?.t_ms??0,true)}</span></div><div className="table-scroll"><table><thead><tr><th>스트림</th><th>기록 수</th><th>중앙 수신률</th><th>기대 주기</th><th>관측 나이</th><th>상태</th></tr></thead><tbody>{run.streams.map(s=>{const current=frame?.streams.find(x=>x.name===s.name);return <tr key={s.name}><td>{s.name}</td><td>{s.count}</td><td>{numeric(s.median_hz,1)} Hz</td><td>{numeric(s.target_hz,1)} Hz</td><td>{numeric(current?.age_ms,0)} ms</td><td>{statusLabel[current?.status??'missing']}</td></tr>;})}</tbody></table></div><div className="panel-footnote">수신률 · 전처리 주기 · 모델 추론 주기 · 화면 갱신률은 별도입니다. 화면 갱신은 최대 10Hz입니다.</div></section>
  <div className="diagnostic-columns"><section className="panel"><div className="panel-heading"><h2>입력 무결성 검사</h2><a className="button small" href={`/api/runs/${run.run_id}/inspection.json`} download><Download size={14}/>검사 결과</a></div><div className="inner"><strong>{run.name}</strong><div className="hash-box"><small>{run.input_sha256?'최종 파일 SHA-256':'진행 중인 세션 · 종료 시 파일 해시 확정'}</small><code>{run.input_sha256??run.run_id}</code></div>{run.issues.length>20&&<p>최근 20건 표시 · 전체 {run.excluded_records}건 제외</p>}{run.issues.slice(-20).map((e,i)=><div className="issue" key={i}><CircleAlert size={15}/><div><strong>{e.line}번째 레코드 제외</strong><p>{e.reason}</p></div></div>)}{!run.issues.length&&<p className="inline-success">제외된 레코드가 없습니다.</p>}{run.boundaries.map((b,i)=><p key={i}>{timeLabel(b.t_ms,true)} · epoch {b.epoch} · {b.reason}</p>)}</div></section>
  <section className="panel"><div className="panel-heading"><h2>입력 계약·연결 상태</h2></div><div className="inner"><dl className="compact-meta"><div><dt>입력 종류</dt><dd>{run.source}</dd></div><div><dt>전처리 버전</dt><dd>{frame?.input_contract?.preprocess_version??'raw-v1'}</dd></div><div><dt>지원 신호</dt><dd>{Object.keys(frame?.input_contract?.signals??{}).length}개</dd></div><div><dt>Epoch</dt><dd>{frame?.epoch??0}</dd></div>{run.session&&<><div><dt>수집 상태</dt><dd>{statusLabel[frame?.session?.state??run.session.state]??run.session.state}</dd></div><div><dt>전처리 sequence 누락</dt><dd>{frame?.session?.dropped_sequences??0}</dd></div></>}</dl><p>학습은 외부에서 수행합니다. 모델이 요구하는 특징·단위·주기·전처리 버전을 연결 시 확인합니다.</p></div></section></div>
  <section className="panel"><details className="signal-details"><summary>표시 신호별 유효성 · 결측과 만료 확인</summary><div className="table-scroll"><table><thead><tr><th>신호</th><th>값</th><th>상태</th><th>관측 나이</th><th>사유</th></tr></thead><tbody>{Object.entries(frame?.values??{}).map(([key,s])=><tr key={key}><td>{s.label}</td><td>{numeric(s.value)} {s.unit}</td><td>{statusLabel[s.status]}</td><td>{numeric(s.age_ms,1)} ms</td><td>{s.reason||'—'}</td></tr>)}</tbody></table></div></details></section>
  <section className="panel"><div className="panel-heading"><h2>분석 연결과 화면 반영</h2><button className="button small" onClick={connectAI}><Plug size={14}/>AI API 설정</button></div><div className="inner integration-grid"><div><small>화면 반영 확인 왕복 p95</small><strong>{numeric(metrics?.render_ack.p95_ms,1)} <span>ms</span></strong><p>{metrics?.render_ack.samples??0}개 표본</p></div><div><small>결과 수신 → 화면 확인 p95</small><strong>{numeric(metrics?.result_receipt_to_render_ack.p95_ms,1)} <span>ms</span></strong><p>{metrics?.result_receipt_to_render_ack.samples??0}개 결과 · 목표 500ms 이내</p></div><p className="method-note">{metrics?.method}</p></div></section>
  </div>;
}
