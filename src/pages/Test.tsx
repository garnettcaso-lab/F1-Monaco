import { useState } from 'react'

export default function Test() {
  const [count, setCount] = useState(0)
  return (
    <div style={{ padding: 40, color: 'white' }}>
      <h1>F1 摩纳哥策略模拟器</h1>
      <p>基础渲染测试通过</p>
      <button onClick={() => setCount(c => c + 1)}>点击: {count}</button>
    </div>
  )
}
