function fmt(v) {
  if (v === null || v === undefined) return '';
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(1);
}

function DataGrid({ panel, values, reads, writes, onEdit }) {
  const readSet = new Set(reads || []);
  const writeSet = new Set(writes || []);
  const editable = panel.kind === 'input';

  return (
    <div className="data-block">
      <div className="data-label">{panel.label}</div>
      <div
        className="data-grid"
        style={{ gridTemplateColumns: `repeat(${panel.cols}, minmax(1.6rem, 1fr))` }}
      >
        {Array.from({ length: panel.rows * panel.cols }, (_, idx) => {
          const cls =
            'cell' +
            (readSet.has(idx) ? ' read' : '') +
            (writeSet.has(idx) ? ' write' : '');
          if (editable) {
            return (
              <input
                key={idx}
                className={cls}
                value={fmt(values[idx])}
                onChange={(e) => onEdit(panel.key, idx, e.target.value)}
              />
            );
          }
          return (
            <div key={idx} className={cls}>
              {fmt(values[idx])}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function DataPanels({ panels, data, reads, writes, onEdit }) {
  return (
    <div className="panel data-panels">
      <h2>Data</h2>
      <p className="hint">Blue = read this step · green = written. Edit inputs to recompute.</p>
      {panels.map((p) => (
        <DataGrid
          key={p.key}
          panel={p}
          values={data[p.key] || []}
          reads={reads[p.key]}
          writes={writes[p.key]}
          onEdit={onEdit}
        />
      ))}
    </div>
  );
}
