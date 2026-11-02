import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'

import About from './pages/About'
import Compare from './pages/Compare'
import Home from './pages/Home'
import Samples from './pages/Samples'

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="site-header">
          <a className="brand" href="/">
            <span className="brand-mark">▮</span> DEFECTLENS
          </a>
          <nav className="site-nav" aria-label="主导航">
            <NavLink to="/" end>
              推理
            </NavLink>
            <NavLink to="/samples">示例库</NavLink>
            <NavLink to="/compare">模型对比</NavLink>
            <NavLink to="/about">方法论</NavLink>
          </nav>
          <span className="status-chip">resnet18-int8</span>
        </header>
        <main className="site-main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/samples" element={<Samples />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <footer className="site-footer">
          <span>DefectLens · 钢表面缺陷检测 · 碳意识模型选型</span>
          <span>
            <a
              href="https://github.com/taokouzi5-lab/steel-defect-lab"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            {' · '}
            <a
              href="https://github.com/taokouzi5-lab/mace-f"
              target="_blank"
              rel="noopener noreferrer"
            >
              mace-f
            </a>
          </span>
        </footer>
      </div>
    </BrowserRouter>
  )
}

export default App
