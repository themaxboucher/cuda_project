export default function TopBar({
  kernels,
  kernelId,
  onKernel,
  kernel,
  params,
  onParam,
  playback,
}) {
  const { index, total, playing, onPlay, onStep, onScrub, speed, onSpeed } = playback;
  return (
    <header className="topbar">
      <div className="topbar-row">
        <div className="brand">CUDA Kernel Visualizer</div>
        <select className="kernel-select" value={kernelId} onChange={(e) => onKernel(e.target.value)}>
          {kernels.map((k) => (
            <option key={k.id} value={k.id}>
              {k.name}
            </option>
          ))}
        </select>

        <div className="controls">
          <button onClick={() => onStep(-1)} disabled={index <= 0} title="Step back">
            ⏮
          </button>
          <button onClick={onPlay} className="play" title="Play / pause">
            {playing ? '⏸' : '▶'}
          </button>
          <button onClick={() => onStep(1)} disabled={index >= total - 1} title="Step forward">
            ⏭
          </button>
          <input
            className="scrub"
            type="range"
            min={0}
            max={total - 1}
            value={index}
            onChange={(e) => onScrub(Number(e.target.value))}
          />
          <span className="step-count">
            {index + 1}/{total}
          </span>
          <label className="speed">
            speed
            <input
              type="range"
              min={1}
              max={5}
              value={speed}
              onChange={(e) => onSpeed(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="topbar-row params">
        {kernel.params.map((p) => (
          <label key={p.key} className="param">
            {p.label}: <b>{params[p.key]}</b>
            <input
              type="range"
              min={p.min}
              max={p.max}
              value={params[p.key]}
              onChange={(e) => onParam(p.key, Number(e.target.value))}
            />
          </label>
        ))}
      </div>
    </header>
  );
}
