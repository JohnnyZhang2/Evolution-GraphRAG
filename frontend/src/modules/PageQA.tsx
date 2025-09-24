import { useRef, useState, useEffect } from 'react'
import { Button, Checkbox, Input, Space, Spin, message, Switch, Slider, Tag, Divider, InputNumber, Drawer, Avatar, Typography } from 'antd'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github.css'
const { TextArea } = Input

const API = import.meta.env.VITE_API_BASE || ''

type ChatMsg = {role:'user'|'assistant', content:string, createdAt?: number, sources?: Array<{id:string, rank?:number, reason?:string, score?:number}>}

export function PageQA(){
  const qRef = useRef<any>(null)
  const [ans, setAns] = useState('')
  const [stream] = useState(true)
  const [sources, setSources] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const elapsedTimerRef = useRef<number | null>(null)
  const [history, setHistory] = useState<Array<ChatMsg>>([])
  const [typewriter] = useState(true)
  const [twSpeed] = useState(35) // ms per tick
  const pendingRef = useRef<string>('')
  const twRef = useRef<number | null>(null)
  const [cfg, setCfg] = useState<any>(null)
  // 上下文窗口控制
  const [useContextWin, setUseContextWin] = useState(true)
  const [ctxTurns, setCtxTurns] = useState<number>(3) // 发送最近 N 轮（user+assistant 视为一轮）
  // 最近一次问题（用于抽屉内高亮）
  const [lastQ, setLastQ] = useState<string>('')
  // 会话标题
  const [convTitle, setConvTitle] = useState<string>('未命名对话')
  // 聊天窗口高度动态计算
  const chatRef = useRef<HTMLDivElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [chatHeight, setChatHeight] = useState<number>(360)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  function updateChatHeight(){
    const el = chatRef.current
    if(!el) return
    try{
      const rect = el.getBoundingClientRect()
      const top = rect.top
      const paddingBottom = 24
      const available = Math.round(window.innerHeight - top - paddingBottom)
      let bottomH = 0
      if(bottomRef.current){
        try{ bottomH = Math.ceil(bottomRef.current.getBoundingClientRect().height) }catch{}
      }
      const h = Math.max(240, available - bottomH)
      setChatHeight(h)
    }catch{}
  }
  // 初始化与监听窗口/布局变化
  useEffect(()=>{
    updateChatHeight()
    const onResize = ()=> updateChatHeight()
    window.addEventListener('resize', onResize)
    const obs = new ResizeObserver(()=> updateChatHeight())
    try{
      obs.observe(document.documentElement)
      if(containerRef.current) obs.observe(containerRef.current)
      if(chatRef.current && chatRef.current.parentElement) obs.observe(chatRef.current.parentElement)
      if(bottomRef.current) obs.observe(bottomRef.current)
    }catch{}
    // 初次渲染后若布局仍在抖动（字体/图片/异步控件），用短 raf settle
    let raf = 0 as number | 0
    let ticks = 0
    const tick = ()=>{
      updateChatHeight()
      ticks += 1
      if(ticks < 12){ raf = requestAnimationFrame(tick) }
    }
    raf = requestAnimationFrame(tick)
    return ()=>{
      window.removeEventListener('resize', onResize)
      try{ obs.disconnect() }catch{}
      if(raf) cancelAnimationFrame(raf as number)
    }
  }, [])
  // 引用详情抽屉（state 提前声明，供依赖使用）
  const [openRef, setOpenRef] = useState<{id:string, rank?:number} | null>(null)
  const [refDetail, setRefDetail] = useState<any>(null)
  const [refLoading, setRefLoading] = useState(false)
  const refCache = useRef<Map<string, any>>(new Map())
  // 关键 UI 状态变化后也触发一次高度计算
  useEffect(()=>{ const t = setTimeout(updateChatHeight, 0); return ()=> clearTimeout(t) }, [cfg, useContextWin, ctxTurns, openRef, loading])
  // 新内容追加时，自动滚动到底部
  useEffect(()=>{
    const el = chatRef.current
    if(!el) return
    el.scrollTop = el.scrollHeight
  }, [ans, history, loading])

  async function openRefDetail(src: {id:string, rank?:number} | undefined){
    if(!src) return
    setOpenRef(src)
    setRefDetail(null)
    const id = src.id
    if(!id) return
    const cached = refCache.current.get(id)
    if(cached){ setRefDetail(cached); return }
    setRefLoading(true)
    try{
      const r = await fetch(`${API}/docs/chunk?id=${encodeURIComponent(id)}`)
      if(!r.ok){ throw new Error(`${r.status} ${r.statusText}`) }
      const j = await r.json()
      refCache.current.set(id, j)
      setRefDetail(j)
    }catch(e:any){
      message.error(e?.message || '加载引用失败')
    }finally{
      setRefLoading(false)
    }
  }

  function formatTime(ts?: number){
    if(!ts) return ''
    try{
      const d = new Date(ts)
      const y = d.getFullYear()
      const m = String(d.getMonth()+1).padStart(2,'0')
      const da = String(d.getDate()).padStart(2,'0')
      const hh = String(d.getHours()).padStart(2,'0')
      const mm = String(d.getMinutes()).padStart(2,'0')
      return `${y}-${m}-${da} ${hh}:${mm}`
    }catch{ return '' }
  }

  function parseStreamSourcesAndTrim(text: string){
    // 解析流式末尾 [SOURCES] 列表，返回 {text, sources}
    const marker = "[SOURCES]"
    const idx = text.lastIndexOf(marker)
    if(idx === -1) return {text, sources: [] as any[]}
    const head = text.slice(0, idx).trimEnd()
    const tail = text.slice(idx).split(/\n/)
    // 逐行解析 "- 1. <id> (rank=..., reason=..., score=...)"
    const list: Array<{id:string, rank?:number, reason?:string, score?:number}> = []
    for(const line of tail){
      const m = /^-\s*(\d+)\.\s+(\S+)/.exec(line)
      if(m){
        const id = m[2]
        // 优先从括号中提取 rank= 数值（与回答中的 [S#] 对应），没有则退回枚举序号
        const mr = /rank\s*=\s*(\d+)/i.exec(line)
        const rank = mr? Number(mr[1]) : Number(m[1])
        const mscore = /score\s*=\s*([\d.]+)/i.exec(line)
        const mreason = /reason\s*=\s*([^,)]+)/i.exec(line)
        list.push({rank, id, score: mscore? Number(mscore[1]) : undefined, reason: mreason? mreason[1] : undefined})
      }
    }
    return {text: head, sources: list}
  }

  function renderWithCitations(
    content: string,
    sources?: Array<{id:string, rank?:number, reason?:string, score?:number}>,
    onRefClick?: (src: {id:string, rank?:number}) => void
  ){
    const byRank = new Map<number, {id:string, rank?:number, reason?:string, score?:number}>()
    ;(sources||[]).forEach(s=>{ if(typeof s.rank === 'number') byRank.set(s.rank, s) })
    const out: Array<JSX.Element> = []
    const pushMd = (md: string, key: string)=> (
      <ReactMarkdown
        key={key}
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{ a: (props)=> <a {...props} target="_blank" rel="noreferrer"/> }}
      >
        {md}
      </ReactMarkdown>
    )
    // 在非代码区域内替换 [S#]，并避免替换 markdown 链接的锚文本（[S#](...)）
    const processPlain = (txt: string, baseKey: string) => {
      const re = /\[S(\d+)\](?!\()/g
      let last = 0
      let m: RegExpExecArray | null
      while((m = re.exec(txt))){
        const start = m.index
        if(start>last){
          out.push(pushMd(txt.slice(last, start), `md-${baseKey}-${last}-${start}`))
        }
        const rank = Number(m[1])
        const src = byRank.get(rank)
        const label = `S${rank}`
        out.push(
          <Tag
            key={`ref-${baseKey}-${start}`}
            color="purple"
            style={{cursor: src? 'pointer':'default', margin:'0 2px'}}
            onClick={()=> src && onRefClick && onRefClick(src)}
          >
            [{label}]
          </Tag>
        )
        last = start + m[0].length
      }
      if(last < txt.length){
        out.push(pushMd(txt.slice(last), `md-${baseKey}-tail-${last}`))
      }
    }
    // 分段处理：代码块（``` ```）与行内代码（`...`）保持原样，其余替换 [S#]
    let i = 0
    let seg = 0
    while(i < content.length){
      const fenceStart = content.indexOf('```', i)
      const endPos = fenceStart === -1 ? content.length : fenceStart
      // 处理 fence 前的普通区域，先按行内代码再拆
      const normal = content.slice(i, endPos)
      if(normal){
        const inlineParts = normal.split(/(`[^`]*`)/)
        inlineParts.forEach((p, idx)=>{
          if(p.startsWith('`') && p.endsWith('`')){
            out.push(pushMd(p, `inline-${seg}-${idx}`))
          }else if(p){
            processPlain(p, `${seg}-${idx}`)
          }
        })
      }
      if(fenceStart === -1) break
      // 处理代码块
      const fenceEnd = content.indexOf('```', fenceStart + 3)
      const close = fenceEnd === -1 ? content.length : fenceEnd + 3
      const codeBlock = content.slice(fenceStart, close)
      out.push(pushMd(codeBlock, `fence-${seg}`))
      i = close
      seg += 1
    }
    return <div>{out}</div>
  }

  function startTimer(){
    const t0 = performance.now()
    if(elapsedTimerRef.current){ window.clearInterval(elapsedTimerRef.current) }
    elapsedTimerRef.current = window.setInterval(()=>{
      setElapsed(((performance.now() - t0) / 1000))
    }, 100)
  }
  function stopTimer(){
    if(elapsedTimerRef.current){ window.clearInterval(elapsedTimerRef.current); elapsedTimerRef.current = null }
  }

  function ensureTypeTimer(){
    if(twRef.current) return
    twRef.current = window.setInterval(()=>{
      const chunkSize = 2 // characters per tick
      if(pendingRef.current.length > 0){
        const take = pendingRef.current.slice(0, chunkSize)
        pendingRef.current = pendingRef.current.slice(chunkSize)
        setAns((prev: string) => prev + take)
      }else{
        // 若没有待写内容且不在加载中，则停止
        if(!loading){
          if(twRef.current){ window.clearInterval(twRef.current); twRef.current = null }
        }
      }
    }, Math.max(10, twSpeed))
  }
  function stopTypeTimer(){
    if(twRef.current){ window.clearInterval(twRef.current); twRef.current = null }
  }
  async function ask(opts?: { qOverride?: string, noUserEcho?: boolean }){
    const rawQ = qRef.current?.resizableTextArea?.textArea?.value?.trim?.()
    const q = (opts?.qOverride ?? rawQ)?.trim?.()
    if(!q) return
    setLastQ(q)
    setAns(''); setSources([]); setElapsed(0); setLoading(true)
    // 预先加载一次配置快照用于显示摘要（非阻塞）
    if(!cfg){
      fetch(`${API}/config`).then(r=>r.json()).then(setCfg).catch(()=>{})
    }
    startTimer()
    // 在外层声明，供 finally 使用
    let finalSources: any[] = []
    try{
      const controller = new AbortController()
      controllerRef.current = controller
      // 发送历史（不包含本次问题）。按上下文窗口限制裁剪。
      const histBefore = history
      const histToSend = useContextWin ? histBefore.slice(-Math.max(0, ctxTurns*2)) : []
      const payloadBase: any = { history: histToSend, question: q }
      // 先将本轮用户消息放入聊天窗口中（重试时可选择不重复插入）
      if(!opts?.noUserEcho){
        setHistory(h => [...h, {role:'user', content: q, createdAt: Date.now()}])
      }
      if(stream){
        const r = await fetch(`${API}/query`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({...payloadBase, stream:true}), signal: controller.signal})
        if(!r.ok || !r.body){
          const txt = await r.text().catch(()=> `${r.status} ${r.statusText}`)
          throw new Error(txt)
        }
        const reader = (r.body as ReadableStream).getReader()
        const dec = new TextDecoder()
        if(typewriter){ pendingRef.current = '' ; ensureTypeTimer() }
        let raw = ''
        // 非打字机模式下，按帧节流刷新，避免频繁 setState 造成阻塞
        let viewBuf = ''
        let raf = 0 as number | 0
        const scheduleFlush = ()=>{
          if(typewriter) return
          if(raf) return
          raf = requestAnimationFrame(()=>{
            if(viewBuf){
              setAns((prev: string) => prev + viewBuf)
              viewBuf = ''
            }
            raf = 0 as any
          })
        }
        while(true){
          const {value, done} = await reader.read()
          if(done) break
          const piece = dec.decode(value, {stream:true})
          if(!piece) continue
          raw += piece
          if(typewriter){
            pendingRef.current += piece
          }else{
            viewBuf += piece
            scheduleFlush()
          }
        }
        // flush decoder tail
        const tail = dec.decode()
        if(tail){
          raw += tail
          if(typewriter){ pendingRef.current += tail } else { viewBuf += tail; }
        }
        // 确保最后一帧刷新完成
        if(!typewriter && viewBuf){ setAns((prev:string)=> prev + viewBuf); viewBuf = '' }
        // 流结束：解析 [SOURCES] 列表，去掉尾部引用说明，填充 sources
        const parsed = parseStreamSourcesAndTrim(raw)
        finalSources = parsed.sources
        setSources(parsed.sources)
        // 若当前答案包含 [SOURCES] 段，去掉它
        if(parsed.text !== raw){
          if(typewriter){
            // 直接替换为最终文本
            pendingRef.current = ''
            setAns(parsed.text)
          }else{
            setAns(parsed.text)
          }
        }
      }else{
        const r = await fetch(`${API}/query`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({...payloadBase, stream:false}), signal: controller.signal})
        if(!r.ok){
          const err = await r.json().catch(()=>({message:`${r.status} ${r.statusText}`}))
          throw new Error(err.message||err.detail||`${r.status}`)
        }
        const j = await r.json()
        const full = j.answer || JSON.stringify(j,null,2)
        if(typewriter){
          setAns('')
          pendingRef.current = full
          ensureTypeTimer()
        }else{
          setAns(full)
        }
        // 优先使用 references（包含精确 label=S# 与 id 的映射）
        if(Array.isArray(j.references) && j.references.length){
          finalSources = j.references.map((ref:any)=>{
            const m = /^S(\d+)$/.exec(ref.label || '')
            const rank = m? Number(m[1]) : undefined
            return {id: ref.id, rank, reason: ref.reason, score: ref.score}
          })
        }else{
          finalSources = j.sources||[]
        }
        setSources(finalSources)
      }
    }catch(e:any){
      if(e?.name === 'AbortError'){
        message.info('已停止生成')
      }else{
        message.error(e?.message || '请求失败')
      }
    }finally{
      setLoading(false)
      stopTimer()
      stopTypeTimer()
      controllerRef.current = null
      // 将本轮回答加入历史（用户消息已在 ask 开始时入列）
      const finalA = (typewriter ? (pendingRef.current || ans) : ans) || ''
      // 清空 pending，避免下一轮误用
      pendingRef.current = ''
      setHistory(h => [...h, {role:'assistant', content: finalA, createdAt: Date.now(), sources: (finalSources && finalSources.length)? finalSources : undefined}])
      // 若会话标题仍为默认，则用第一条问题生成标题
      setConvTitle(t => (t && t !== '未命名对话') ? t : (lastQ ? (lastQ.length>24? (lastQ.slice(0,24)+'…') : lastQ) : '未命名对话'))
    }
  }
  function retryLast(){
    if(loading) return
    // 仅当最后一条是 assistant 时可重试
    setHistory(h => {
      if(h.length===0) return h
      const last = h[h.length-1]
      if(last.role !== 'assistant') return h
      // 移除最后一条 assistant，并触发重试（不重复回显用户）
      const newHist = h.slice(0, -1)
      // 触发异步 ask（使用 setTimeout 以确保 state 更新先生效）
      setTimeout(()=> ask({ qOverride: lastQ || (newHist[newHist.length-1]?.content || ''), noUserEcho: true }), 0)
      return newHist
    })
  }
  // --- 引用抽屉高亮函数 ---
  function escapeReg(s:string){ return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') }
  function buildTerms(q: string, ents: string[]|undefined){
    const t: string[] = []
    if(q){
      const ws = q.match(/[A-Za-z0-9_]{3,}/g) || []
      t.push(...ws)
      const cs = q.match(/[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]{2,}/g) || []
      t.push(...cs)
    }
    if(Array.isArray(ents)){
      for(const e of ents){ if(e && e.length>=2) t.push(e) }
    }
    const uniq = Array.from(new Set(t.map(s=>s.trim()))).filter(Boolean)
    uniq.sort((a,b)=> b.length - a.length)
    return uniq.slice(0, 12)
  }
  function renderHighlighted(text: string, terms: string[]){
    if(!text || terms.length===0) return <span>{text}</span>
    const pattern = new RegExp(terms.map(escapeReg).join('|'), 'gi')
    const parts: Array<JSX.Element|string> = []
    let lastIndex = 0
    let m: RegExpExecArray | null
    while((m = pattern.exec(text))){
      const start = m.index
      if(start>lastIndex) parts.push(text.slice(lastIndex, start))
      const hit = text.slice(start, start + m[0].length)
      parts.push(<mark key={`hl-${start}`} style={{background:'#fff59d'}}>{hit}</mark>)
      lastIndex = start + m[0].length
    }
    if(lastIndex < text.length) parts.push(text.slice(lastIndex))
    return <span>{parts}</span>
  }
  function stop(){
    if(controllerRef.current){
      try{ controllerRef.current.abort() }catch{}
      controllerRef.current = null
    }
  }
  async function copyAns(){
    try{ await navigator.clipboard.writeText(ans); message.success('回答已复制') }catch{ message.error('复制失败') }
  }
  function clearAll(){ setAns(''); setSources([]); setHistory([]); if(qRef.current){ qRef.current.resizableTextArea.textArea.value = '' } }
  function newConversation(){
    clearAll()
    setConvTitle('未命名对话')
  }

  function cfgSummary(c:any){
    if(!c) return null
    const keys = [
      ['top_k', c.top_k],
      ['expand_hops', c.expand_hops],
      ['bm25_enabled', c.bm25_enabled],
      ['graph_rank_enabled', c.graph_rank_enabled],
      ['rerank_enabled', c.rerank_enabled]
    ]
    return (
      <Space wrap>
        {keys.map(([k,v])=> (
          <Tag key={String(k)} color={String(v)==='true'||v===true? 'green':'blue'}>{k}:{String(v)}</Tag>
        ))}
      </Space>
    )
  }
  // 渲染聊天窗口（历史 + 正在生成的回答临时气泡）
  const renderMessages: Array<ChatMsg> = [...history]
  if(loading || ans){
    renderMessages.push({role:'assistant', content: ans, sources})
  }

  return (
  <div className="card" ref={containerRef}>
      <h3>Evolution</h3>
      <div style={{display:'flex', alignItems:'center', gap:12, margin:'6px 0 10px'}}>
        <span style={{opacity:0.7}}>会话标题：</span>
        <Input size="small" style={{maxWidth:360}} value={convTitle} onChange={e=>setConvTitle(e.target.value)} placeholder="输入会话标题" />
      </div>
      <div style={{marginBottom:8}}>
        <Space size={16} wrap>
          <span>
            上下文：<Switch size="small" checked={useContextWin} onChange={setUseContextWin} disabled={loading} />
          </span>
          <span>
            窗口轮数：<InputNumber min={0} max={10} value={ctxTurns} onChange={(v)=>setCtxTurns((v||0) as number)} disabled={loading || !useContextWin} />
          </span>
        </Space>
      </div>

      {cfg && (
        <div style={{marginBottom:8}}>
          <span style={{marginRight:8, opacity:0.7}}>配置摘要：</span>
          {cfgSummary(cfg)}
        </div>
      )}

      <div ref={chatRef} style={{border:'1px solid #eee', borderRadius:8, padding:12, height: chatHeight, minHeight:240, overflow:'auto', background:'linear-gradient(180deg, #ffffff 0%, #f7faff 100%)', marginBottom:8}}>
        <div style={{maxWidth:900, margin:'0 auto'}}>
        {renderMessages.length===0 ? (
          <div style={{opacity:0.6, textAlign:'center', padding:'12px 0'}}>开始你的第一条对话吧～</div>
        ) : renderMessages.map((m,i)=> {
          const isUser = m.role==='user'
          return (
            <div key={i} style={{
              display:'flex',
              gap:12,
              alignItems:'flex-start',
              flexDirection: isUser? 'row-reverse' : 'row',
              margin:'14px 0'
            }}>
              <div style={{display:'flex', flexDirection:'column', alignItems: isUser? 'flex-end':'flex-start', minWidth:36}}>
                <Avatar size={32} style={{ backgroundColor: isUser? '#1677ff' : '#52c41a' }}>
                  {isUser? '我' : 'E'}
                </Avatar>
                <div style={{fontSize:12, color:'#999', marginTop:4}}>{isUser? '我' : 'Evolution'}</div>
                <div style={{fontSize:11, color:'#bbb'}}>{formatTime(m.createdAt)}</div>
              </div>
              <div style={{
                maxWidth:'85%',
                background: isUser? '#e6f4ff' : '#ffffff',
                border: '1px solid #eee',
                padding:'10px 12px',
                borderRadius: isUser? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                whiteSpace:'pre-wrap',
                boxShadow:'0 2px 6px rgba(0,0,0,0.05)',
                position:'relative'
              }}>
                {isUser ? (
                  // 用户消息纯文本展示
                  <div style={{lineHeight:1.6}}>{m.content}</div>
                ) : (
                  m.content ? (
                    <div style={{lineHeight:1.7}}>
                      {renderWithCitations(m.content, m.sources, openRefDetail)}
                    </div>
                  ) : (
                    loading ? <><Spin size="small" /> <span style={{marginLeft:6, opacity:0.7}}>正在生成…</span></> : ''
                  )
                )}
                {/* 操作按钮 */}
                <div style={{marginTop:6, display:'flex', gap:12, justifyContent: isUser? 'flex-end':'flex-start', opacity:0.8}}>
                  <Button size="small" type="text" onClick={async()=>{ try{ await navigator.clipboard.writeText(m.content); message.success('已复制'); }catch{ message.error('复制失败') } }}>复制</Button>
                  {(!isUser && i === history.length-1 && !loading) && (
                    <Button size="small" type="text" onClick={retryLast}>重试</Button>
                  )}
                </div>
              </div>
            </div>
          )
        })}
        </div>
      </div>

      <div
        ref={bottomRef}
        style={{
          marginTop:8,
          position:'sticky',
          bottom:0,
          background:'#fff',
          paddingTop:8,
          zIndex:10,
          borderTop:'1px solid #eee',
          boxShadow:'0 -2px 8px rgba(0,0,0,0.04)'
        }}
      >
        <TextArea
          ref={qRef}
          placeholder="请输入你的问题...（Enter 发送，Shift+Enter 换行）"
          autoSize={{minRows:2, maxRows:6}}
          disabled={loading}
          onKeyDown={(e)=>{
            if(e.key==='Enter' && !e.shiftKey){
              e.preventDefault()
              if(!loading) ask()
            }
          }}
        />
        <Space style={{marginTop:8, flexWrap:'wrap'}}>
          <Button type="primary" onClick={()=>ask()} loading={loading} disabled={loading}>发送</Button>
          {loading && <Button danger onClick={stop}>停止生成</Button>}
          {!loading && ans && <Button onClick={copyAns}>复制最后回答</Button>}
          {!loading && (ans || history.length>0 || sources.length>0) && <Button onClick={newConversation}>新对话</Button>}
          {loading && <span style={{opacity:0.7}}>已用时 {elapsed.toFixed(1)}s</span>}
        </Space>

        <div className="row" style={{marginTop:12}}>
          <div style={{flex:1}}>
            <h4>引用</h4>
            <div className="list">
              {sources.map((s:any,i:number)=> (
                <div key={i} style={{cursor:'pointer'}} onClick={()=>openRefDetail(s)}>
                  <span className="pill">S{s.rank||i+1}</span> <code>{s.id}</code>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <Drawer
        open={!!openRef}
        width={720}
        title={openRef ? `引用 ${openRef.rank ? 'S'+openRef.rank : ''} — ${openRef.id}` : '引用详情'}
        onClose={()=>{ setOpenRef(null); setRefDetail(null) }}
      >
        {refLoading ? (
          <div style={{padding:'24px 0'}}><Spin /> <span style={{marginLeft:8, opacity:0.7}}>加载中…</span></div>
        ) : (
          refDetail ? (
            <div>
              <div style={{marginBottom:8}}>
                <Tag color="blue">Source</Tag> <code>{refDetail.source || openRef?.id}</code>
              </div>
              <div style={{whiteSpace:'pre-wrap', border:'1px solid #eee', borderRadius:8, padding:12, background:'#fff'}}>
                {renderHighlighted(refDetail.text || '(无内容)', buildTerms(lastQ, refDetail.entities))}
              </div>
              {Array.isArray(refDetail.entities) && refDetail.entities.length>0 && (
                <div style={{marginTop:12}}>
                  <div style={{marginBottom:6, opacity:0.8}}>实体：</div>
                  <Space wrap>
                    {refDetail.entities.map((e:string, idx:number)=> <Tag key={idx}>{e}</Tag>)}
                  </Space>
                </div>
              )}
            </div>
          ) : (
            openRef ? <div style={{opacity:0.7}}>未找到更多信息，ID: <code>{openRef.id}</code></div> : null
          )
        )}
      </Drawer>
    </div>
  )
}
