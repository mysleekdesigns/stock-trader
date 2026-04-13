import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Strategies from './pages/Strategies'
import Backtest from './pages/Backtest'
import Models from './pages/Models'
import Orders from './pages/Orders'
import Settings from './pages/Settings'
import ORBScanner from './pages/ORBScanner'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/strategies" element={<Strategies />} />
        <Route path="/orb" element={<ORBScanner />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/models" element={<Models />} />
        <Route path="/orders" element={<Orders />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App
