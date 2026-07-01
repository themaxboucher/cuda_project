export default function CodePanel({ source, activeLine, blurb }) {
  return (
    <div className="panel code-panel">
      <h2>Kernel code</h2>
      {blurb && <p className="blurb">{blurb}</p>}
      <pre>
        {source.map((line, i) => (
          <div key={i} className={'code-line' + (i === activeLine ? ' active' : '')}>
            <span className="ln">{i + 1}</span>
            <code>{line || ' '}</code>
          </div>
        ))}
      </pre>
    </div>
  );
}
