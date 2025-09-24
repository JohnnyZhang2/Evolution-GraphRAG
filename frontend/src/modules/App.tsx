import { useState } from 'react'
import { PageQA } from './PageQA'
import { PageDocs } from './PageDocs'
import { PageLogs } from './PageLogs'
import { PageConfig } from './PageConfig'

type Tab = 'qa'|'docs'|'logs'|'config'

export function App(){
  const [tab, setTab] = useState<Tab>('qa')
  return (
    <div>
      <header className="header">
        <h1>Evolution GraphRAG 控制台</h1>
        <nav>
          <button onClick={()=>setTab('qa')}>问答</button>
          <button onClick={()=>setTab('docs')}>文档管理</button>
          <button onClick={()=>setTab('logs')}>问答日志</button>
          <button onClick={()=>setTab('config')}>配置</button>
        </nav>
      </header>
      <main className="main">
        {tab==='qa' && <PageQA/>}
        {tab==='docs' && <PageDocs/>}
        {tab==='logs' && <PageLogs/>}
        {tab==='config' && <PageConfig/>}
      </main>
    </div>
  )
}
