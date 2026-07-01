function fmt(v) {
  if (v === null || v === undefined) return '·';
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(1);
}

function Tile({ label, values, TILE, active }) {
  return (
    <div className="shared-tile">
      <div className="data-label">{label}</div>
      <div
        className={'data-grid' + (active ? ' tile-active' : '')}
        style={{ gridTemplateColumns: `repeat(${TILE}, minmax(1.6rem, 1fr))` }}
      >
        {Array.from({ length: TILE * TILE }, (_, i) => (
          <div key={i} className={'cell' + (values[i] === null ? ' empty' : ' filled')}>
            {fmt(values[i])}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SharedPanel({ shared, barrier }) {
  return (
    <div className="panel shared-panel">
      <h2>Shared memory {shared ? <span className="dim">block {shared.block}</span> : null}</h2>
      {barrier ? (
        <div className="barrier">
          🚧 __syncthreads() barrier — {barrier.arrived}/{barrier.total} threads arrived
        </div>
      ) : (
        <div className="barrier idle">threads running (no barrier)</div>
      )}
      {shared ? (
        <div className="shared-tiles">
          <Tile label="aTile" values={shared.aTile} TILE={shared.TILE} active={shared.highlight === 'compute' || shared.highlight === 'load'} />
          <Tile label="bTile" values={shared.bTile} TILE={shared.TILE} active={shared.highlight === 'compute' || shared.highlight === 'load'} />
        </div>
      ) : (
        <p className="hint">Shared tiles appear while a block is running.</p>
      )}
    </div>
  );
}
