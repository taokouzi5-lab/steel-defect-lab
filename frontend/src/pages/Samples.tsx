function Samples() {
  return (
    <section aria-labelledby="samples-heading">
      <h1 id="samples-heading" className="section-title">
        <span className="kicker">Test Set</span>
        示例库
      </h1>
      <p style={{ color: 'var(--text-dim)', maxWidth: '56ch' }}>
        NEU-DET 测试集样本（未参与训练）。点击任意样本直接推演。
      </p>
      <div className="card" style={{ marginTop: '2rem' }}>
        <p className="mono" style={{ color: 'var(--text-dim)' }}>
          [示例卡片将在下一步实现]
        </p>
      </div>
    </section>
  )
}

export default Samples
