import { useEffect, useMemo, useRef, useState } from 'react';
import { kernels, getKernel, defaultParams } from './kernels/index.js';
import TopBar from './components/TopBar.jsx';
import CodePanel from './components/CodePanel.jsx';
import DataPanels from './components/DataPanels.jsx';
import SharedPanel from './components/SharedPanel.jsx';
import Scene from './components/Scene.jsx';
import { GLOBAL_LATENCY, SHARED_LATENCY, BASE_COST } from './sim/memory.js';

const SPEED_DELAY = { 1: 1600, 2: 1100, 3: 750, 4: 450, 5: 250 };
// A step costing one global round trip (2 × GLOBAL_LATENCY) plays for one full
// base tick; cheaper steps go proportionally faster, floored so they stay visible.
const COST_REF = 2 * GLOBAL_LATENCY;

function stepDelay(speed, cost) {
  const base = SPEED_DELAY[speed];
  return Math.max(base * 0.2, Math.min(base * 6, (base * (cost ?? BASE_COST)) / COST_REF));
}

function MemoryBadge({ memory }) {
  if (memory === 'global') {
    return (
      <span className="mem-badge mem-global">
        🐢 GLOBAL memory · ~{GLOBAL_LATENCY} cycles/access · slow
      </span>
    );
  }
  if (memory === 'shared') {
    return (
      <span className="mem-badge mem-shared">
        ⚡ SHARED memory · ~{SHARED_LATENCY} cycles/access · fast
      </span>
    );
  }
  return <span className="mem-badge mem-none">no memory traffic</span>;
}

export default function App() {
  const [kernelId, setKernelId] = useState(kernels[0].id);
  const kernel = getKernel(kernelId);

  const [params, setParams] = useState(() => defaultParams(kernel));
  const [data, setData] = useState(() => kernel.makeDefaultData(defaultParams(kernel)));

  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(3);

  const trace = useMemo(() => kernel.generateTrace(params, data), [kernel, params, data]);

  // Any regeneration (kernel, params, or edited data) rewinds to the start.
  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [trace]);

  // Autoplay: advance one step per tick until the end.
  useEffect(() => {
    if (!playing) return undefined;
    if (index >= trace.steps.length - 1) {
      setPlaying(false);
      return undefined;
    }
    // Dwell on the CURRENT step for a time proportional to its cycle cost, so
    // global-memory steps visibly take longer than shared-memory ones.
    const id = setTimeout(
      () => setIndex((i) => Math.min(i + 1, trace.steps.length - 1)),
      stepDelay(speed, trace.steps[index].cost),
    );
    return () => clearTimeout(id);
  }, [playing, index, speed, trace]);

  function selectKernel(id) {
    const k = getKernel(id);
    const p = defaultParams(k);
    setKernelId(id);
    setParams(p);
    setData(k.makeDefaultData(p));
  }

  function setParam(key, value) {
    const next = { ...params, [key]: value };
    setParams(next);
    setData(kernel.makeDefaultData(next)); // sizes changed -> reset data to defaults
  }

  function editCell(key, idx, raw) {
    const val = raw === '' ? 0 : Number(raw);
    if (Number.isNaN(val)) return;
    setData((d) => ({ ...d, [key]: d[key].map((v, i) => (i === idx ? val : v)) }));
  }

  function stepBy(delta) {
    setPlaying(false);
    setIndex((i) => Math.max(0, Math.min(trace.steps.length - 1, i + delta)));
  }

  // Keyboard: space = play/pause, arrows = step.
  const stepRef = useRef(stepBy);
  stepRef.current = stepBy;
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.code === 'ArrowRight') {
        stepRef.current(1);
      } else if (e.code === 'ArrowLeft') {
        stepRef.current(-1);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const step = trace.steps[Math.min(index, trace.steps.length - 1)];
  const isTiled = kernel.id === 'matmulTiled';

  return (
    <div className="app">
      <TopBar
        kernels={kernels}
        kernelId={kernelId}
        onKernel={selectKernel}
        kernel={kernel}
        params={params}
        onParam={setParam}
        playback={{
          index,
          total: trace.steps.length,
          playing,
          onPlay: () => setPlaying((p) => !p),
          onStep: stepBy,
          onScrub: (v) => {
            setPlaying(false);
            setIndex(v);
          },
          speed,
          onSpeed: setSpeed,
        }}
      />

      <div className="workspace">
        <CodePanel source={trace.source} activeLine={step.line} blurb={kernel.blurb} />

        <div className="center">
          <Scene trace={trace} step={step} />
          <div className="memory-bar">
            <MemoryBadge memory={step.memory} />
            <span className="gpu-clock">
              GPU clock: <b>{step.cycles.toLocaleString()}</b> / {trace.totalCycles.toLocaleString()} cycles
            </span>
          </div>
          <div className={'status phase-' + step.phase}>
            <span className="phase-tag">{step.phase || '—'}</span>
            {step.caption}
          </div>
        </div>

        <div className="right">
          {isTiled && <SharedPanel shared={step.shared} barrier={step.barrier} />}
          <DataPanels
            panels={trace.dataPanels}
            data={step.data}
            reads={step.reads}
            writes={step.writes}
            onEdit={editCell}
          />
        </div>
      </div>
    </div>
  );
}
