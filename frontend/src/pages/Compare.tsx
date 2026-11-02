function Compare() {
  return (
    <section aria-labelledby="compare-heading">
      <h1 id="compare-heading" className="section-title">
        <span className="kicker">Model Selection</span>
        模型对比
      </h1>
      <p style={{ color: 'var(--text-dim)', maxWidth: '56ch' }}>
        选型标准：精度达标前提下碳效率最优。数据来自训练实测与 INT8 量化门禁。
      </p>
      <div className="card" style={{ marginTop: '2rem' }}>
        <p className="mono" style={{ color: 'var(--text-dim)' }}>
          [决策表将在下一步实现]
        </p>
      </div>
    </section>
  )
}

export default Compare
