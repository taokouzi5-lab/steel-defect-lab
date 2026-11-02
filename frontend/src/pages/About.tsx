function About() {
  return (
    <section aria-labelledby="about-heading">
      <h1 id="about-heading" className="section-title">
        <span className="kicker">Methodology</span>
        方法论
      </h1>
      <p style={{ color: 'var(--text-dim)', maxWidth: '56ch' }}>
        碳成本公式与常量出处、测量边界声明。数据来自 mace-f 框架的实测基准。
      </p>
      <div className="card" style={{ marginTop: '2rem' }}>
        <p className="mono" style={{ color: 'var(--text-dim)' }}>
          [碳方法说明将在下一步实现]
        </p>
      </div>
    </section>
  )
}

export default About
