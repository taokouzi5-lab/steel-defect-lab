function Home() {
  return (
    <section aria-labelledby="home-heading">
      <h1 id="home-heading" style={{ fontSize: 'var(--text-hero)' }}>
        钢表面缺陷检测
        <span className="mono" style={{ color: 'var(--accent)' }}>
          {' '}
          · 碳意识选型
        </span>
      </h1>
      <p style={{ color: 'var(--text-dim)', maxWidth: '56ch', marginTop: '1rem' }}>
        上传一张钢材表面图像，resnet18-int8 在本地 CPU 上完成分类，Grad-CAM
        指出缺陷位置，并告诉你这次推理的碳成本。
      </p>
      <div className="card" style={{ marginTop: '2rem' }}>
        <p className="mono" style={{ color: 'var(--text-dim)' }}>
          [上传区将在下一步实现]
        </p>
      </div>
    </section>
  )
}

export default Home
