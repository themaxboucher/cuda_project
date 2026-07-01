import { useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Instances, Instance, Html } from '@react-three/drei';
import { enumerateBlocks, enumerateThreads, blockKey, threadKey } from '../sim/topology.js';

const CELL = 1.0; // spacing between thread cubes
const GAP = 1.6; // gap between blocks
const CUBE = 0.78; // cube edge length

// Pastel base colors so different blocks are visually distinguishable on a
// light background.
const BLOCK_COLORS = [
  '#8aa0d6', '#b19ad9', '#8ad3bd', '#d6c08a',
  '#d99aad', '#9ad3d9', '#b4d69a', '#d09ad0',
];
const ACTIVE = '#76b900';
const DONE = '#cfe4a0';
const SCENE_BG = '#f6f8fc';

function buildCubes(gridDim, blockDim) {
  const blocks = enumerateBlocks(gridDim);
  const threads = enumerateThreads(blockDim);
  const strideX = blockDim.x * CELL + GAP;
  const strideY = blockDim.y * CELL + GAP;
  const strideZ = blockDim.z * CELL + GAP;

  const cubes = [];
  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];

  for (const b of blocks) {
    for (const t of threads) {
      const pos = [
        b.bx * strideX + t.tx * CELL,
        b.by * strideY + t.ty * CELL,
        b.bz * strideZ + t.tz * CELL,
      ];
      for (let i = 0; i < 3; i++) {
        min[i] = Math.min(min[i], pos[i]);
        max[i] = Math.max(max[i], pos[i]);
      }
      cubes.push({
        key: threadKey(b, t),
        bkey: blockKey(b),
        block: { bx: b.bx, by: b.by, bz: b.bz, index: b.index },
        thread: { tx: t.tx, ty: t.ty, tz: t.tz, tid: t.tid },
        pos,
      });
    }
  }

  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  for (const c of cubes) {
    c.pos = [c.pos[0] - center[0], c.pos[1] - center[1], c.pos[2] - center[2]];
  }
  const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
  return { cubes, span };
}

function ThreadCube({ cube, state, onHover, onUnhover }) {
  let color = BLOCK_COLORS[cube.block.index % BLOCK_COLORS.length];
  let scale = CUBE;
  if (state === 'active') {
    color = ACTIVE;
    scale = CUBE * 1.12;
  } else if (state === 'done') {
    color = DONE;
  }
  return (
    <Instance
      position={cube.pos}
      scale={scale}
      color={color}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(cube);
      }}
      onPointerOut={onUnhover}
    />
  );
}

export default function Scene({ trace, step }) {
  const { gridDim, blockDim } = trace;
  const { cubes, span } = useMemo(() => buildCubes(gridDim, blockDim), [gridDim, blockDim]);
  const [hovered, setHovered] = useState(null);

  // A stable id for the current topology. The instanced mesh must be remounted
  // (not just re-rendered) when the cube COUNT changes, otherwise drei's
  // <Instances> keeps its original buffer size and the scene goes blank.
  const topoKey = `${gridDim.x}x${gridDim.y}x${gridDim.z}-${blockDim.x}x${blockDim.y}x${blockDim.z}`;

  // Drop any stale hover target when the topology changes.
  useEffect(() => {
    setHovered(null);
  }, [topoKey]);

  const activeSet = useMemo(() => new Set(step.activeThreads), [step]);
  const doneSet = useMemo(() => new Set(step.doneThreads), [step]);

  const dist = span * 1.8 + 4;

  return (
    <div className="scene">
      <Canvas camera={{ position: [dist, dist * 0.8, dist], fov: 45 }}>
        <color attach="background" args={[SCENE_BG]} />
        <ambientLight intensity={0.85} />
        <directionalLight position={[10, 15, 10]} intensity={1.0} />
        <directionalLight position={[-8, -4, -6]} intensity={0.35} />
        <Instances key={topoKey} limit={cubes.length} range={cubes.length}>
          <boxGeometry />
          <meshStandardMaterial roughness={0.45} metalness={0.1} toneMapped={false} />
          {cubes.map((cube) => {
            const state = activeSet.has(cube.key)
              ? 'active'
              : doneSet.has(cube.key)
                ? 'done'
                : 'idle';
            return (
              <ThreadCube
                key={cube.key}
                cube={cube}
                state={state}
                onHover={setHovered}
                onUnhover={() => setHovered(null)}
              />
            );
          })}
        </Instances>
        {hovered && (
          <Html position={hovered.pos} className="cube-tip" wrapperClass="cube-tip-wrap">
            <div>
              block ({hovered.block.bx},{hovered.block.by},{hovered.block.bz})
              <br />
              thread ({hovered.thread.tx},{hovered.thread.ty},{hovered.thread.tz})
            </div>
          </Html>
        )}
        <OrbitControls enableDamping makeDefault />
      </Canvas>
      <div className="scene-legend">
        <span><i style={{ background: ACTIVE }} /> executing</span>
        <span><i style={{ background: DONE }} /> done</span>
        <span><i style={{ background: BLOCK_COLORS[0] }} /> pending (color = block)</span>
      </div>
    </div>
  );
}
