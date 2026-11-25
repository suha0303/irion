import type { V02Model } from './local-flight-types';

const families = [
  ['inflight', '비행 중 예측'],
  ['whole-run', '비행 종료 후 판정'],
] as const;

export function modelIds(models: V02Model[], selection: string) {
  if (selection === 'all') return models.map((model) => model.id);
  if (selection === 'inflight' || selection === 'whole-run')
    return models.filter((model) => model.task === selection).map((model) => model.id);
  const model = models.find((entry) => entry.id === selection) ?? models[0];
  return model ? [model.id] : [];
}

export function modelLabel(id: string) {
  const label = id
    .replace(/^(inflight|whole-run)-/, '')
    .replace('lstm-', 'LSTM ')
    .replace('transformer-', 'Transformer ')
    .replace('x', '×');
  return `${id.startsWith('whole-run-') ? '종료 후 판정' : '비행 중 예측'} · ${label}`;
}

export default function ModelSelection({
  models,
  selected,
  onChange,
}: {
  models: V02Model[];
  selected: string;
  onChange: (value: string) => void;
}) {
  const current = modelIds(models, selected);
  return (
    <label className="form-label">
      모델 선택
      <select
        className="text-input"
        value={current.length > 1 ? selected : (current[0] ?? '')}
        disabled={!models.length}
        onChange={(event) => onChange(event.target.value)}
      >
        {!models.length && <option value="">연결된 모델 없음</option>}
        {models.length > 1 && <option value="all">전체 {models.length}종 일괄</option>}
        {families.map(([task, label]) => {
          const entries = models.filter((model) => model.task === task);
          return (
            entries.length > 0 && (
              <optgroup key={task} label={label}>
                {entries.length > 1 && (
                  <option value={task}>
                    {label} {entries.length}종 일괄
                  </option>
                )}
                {entries.map((model) => (
                  <option value={model.id} key={model.id}>
                    {label} · {model.label}
                  </option>
                ))}
              </optgroup>
            )
          );
        })}
      </select>
    </label>
  );
}
