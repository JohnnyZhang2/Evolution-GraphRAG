import { useEffect, useMemo, useRef, useState } from 'react'
import { Table, Button, Input, Checkbox, Modal, Typography, Pagination, Space, Tag, message } from 'antd'
import CytoscapeComponent from 'react-cytoscapejs'
const API = import.meta.env.VITE_API_BASE || ''

export function PageDocs(){
  const [sources, setSources] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [loading, setLoading] = useState(false)
  const pathRef = useRef<HTMLInputElement>(null)
  const [inc, setInc] = useState(true)
  const [refresh, setRefresh] = useState(false)
  const [refreshRel, setRefreshRel] = useState(true)

  async function load(){
    setLoading(true)
    try{
      const skip = (page-1)*pageSize
      const r = await fetch(`${API}/docs/sources?limit=${pageSize}&skip=${skip}`)
      const j = await r.json(); setSources(j.items||[]); setTotal(j.total||0)
    }finally{ setLoading(false) }
  }
  async function ingest(){
    const p = pathRef.current?.value.trim(); if(!p) return message.warning('请输入路径')
    const u = new URL(`${API}/ingest/stream`)
    u.searchParams.set('path', p!)
    u.searchParams.set('incremental', String(inc))
    u.searchParams.set('refresh', String(refresh))
    u.searchParams.set('refresh_relations', String(refreshRel))
    const es = new EventSource(u.toString())
    es.addEventListener('result', (e:any)=>{ try{ const d = JSON.parse(e.data); message.success('完成: '+JSON.stringify(d.result)) }catch{}; load() })
  }
  async function delSource(src:string){
    Modal.confirm({
      title: '确认删除来源?',
      content: src,
      onOk: async () => {
        const u = new URL(`${API}/docs/source`)
        u.searchParams.set('source', src)
        const r = await fetch(u.toString(), {method:'DELETE'})
        if(r.ok) load()
      }
    })
  }

  useEffect(()=>{ load() },[page,pageSize])

  // Source row expand: load chunks table
  const [chunkModal, setChunkModal] = useState<{open:boolean, id?:string, data?:any}>({open:false})
  const [chunkLoading, setChunkLoading] = useState(false)
  const [graph, setGraph] = useState<{nodes:any[], edges:any[]}>({nodes:[], edges:[]})
  async function openChunk(id: string){
    setChunkModal({open:true, id, data:undefined})
    setChunkLoading(true)
    try{
      const r = await fetch(`${API}/docs/chunk?id=${encodeURIComponent(id)}`)
      if(!r.ok){
        const txt = await r.text().catch(()=> `${r.status} ${r.statusText}`)
        message.error(`获取详情失败：${txt}`)
        setChunkModal({open:true, id, data:undefined})
        setGraph({nodes:[], edges:[]})
        return
      }
      const d = await r.json().catch(()=>null)
      if(!d){
        message.error('解析详情失败')
        setGraph({nodes:[], edges:[]})
        return
      }
      setChunkModal({open:true, id, data:d})
      try{
        const gr = await fetch(`${API}/graph/egonet?id=${encodeURIComponent(id)}&depth=1&limit=120`)
        if(gr.ok){
          const g = await gr.json().catch(()=>({nodes:[], edges:[]}))
          setGraph({nodes: Array.isArray(g.nodes)? g.nodes:[], edges: Array.isArray(g.edges)? g.edges:[]})
        }else{
          setGraph({nodes:[], edges:[]})
        }
      }catch{
        setGraph({nodes:[], edges:[]})
      }
    }catch(e:any){
      message.error(e?.message || '加载失败')
      setGraph({nodes:[], edges:[]})
    }finally{
      setChunkLoading(false)
    }
  }

  return (
    <div className="card">
      <h3>文档管理</h3>
      <Space className="row" wrap>
        <Input ref={pathRef as any} placeholder="文件或目录绝对路径" style={{width:420}} />
        <Checkbox checked={inc} onChange={e=>setInc(e.target.checked)}>incremental</Checkbox>
        <Checkbox checked={refresh} onChange={e=>setRefresh(e.target.checked)}>refresh</Checkbox>
        <Checkbox checked={refreshRel} onChange={e=>setRefreshRel(e.target.checked)}>refresh_rel</Checkbox>
        <Button type="primary" onClick={ingest}>开始导入</Button>
      </Space>
      <Table
        rowKey="source"
        loading={loading}
        dataSource={sources}
        columns={[
          {title:'来源', dataIndex:'source', render: (v)=> <Typography.Text code>{v}</Typography.Text>},
          {title:'Chunks', dataIndex:'chunks'},
          {title:'最后更新时间', dataIndex:'updatedAt'},
          {title:'操作', render: (_,r)=> <Space><Button size="small" danger onClick={()=>delSource(r.source)}>删除</Button></Space>}
        ]}
        pagination={false}
        expandable={{
          expandedRowRender: (record)=> <ChunksTable source={record.source} onOpen={openChunk} />,
          rowExpandable: (record)=> record.chunks>0
        }}
      />
      <div style={{display:'flex', justifyContent:'flex-end', marginTop:8}}>
        <Pagination current={page} pageSize={pageSize} total={total} onChange={(p,ps)=>{setPage(p); setPageSize(ps)}} showSizeChanger/>
      </div>

      <Modal open={chunkModal.open} width={980} onCancel={()=>setChunkModal({open:false})} footer={null} title={`Chunk 详情: ${chunkModal.id}`}>
        {chunkLoading && <div style={{padding:'24px 0'}}>加载中…</div>}
        {(!chunkLoading && chunkModal.data) && (
          <div>
            <Space direction="vertical" style={{width:'100%'}} size="middle">
              <div>
                <Typography.Text strong>来源:</Typography.Text> <Typography.Text code>{chunkModal.data.source}</Typography.Text>
              </div>
              <div>
                <Typography.Text strong>实体:</Typography.Text> {(chunkModal.data.entities||[]).map((e:string)=>(<Tag key={e}>{e}</Tag>))}
              </div>
              <div style={{maxHeight:200, overflow:'auto'}}>
                <Typography.Paragraph copyable ellipsis={false}>
                  {chunkModal.data.text}
                </Typography.Paragraph>
              </div>
              <div>
                <Typography.Text strong>局部图谱</Typography.Text>
                {chunkLoading && <Typography.Text type="secondary" style={{marginLeft:8}}>构建中…</Typography.Text>}
                <div style={{height:360, border:'1px solid #eee', borderRadius:8}}>
                  <GraphView data={graph} visible={chunkModal.open} />
                </div>
              </div>
            </Space>
          </div>
        )}
      </Modal>
    </div>
  )
}

function ChunksTable({source,onOpen}:{source:string,onOpen:(id:string)=>void}){
  const [items, setItems] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [loading, setLoading] = useState(false)
  async function load(){
    setLoading(true)
    try{
      const skip = (page-1)*pageSize
      const r = await fetch(`${API}/docs/chunks?source=${encodeURIComponent(source)}&limit=${pageSize}&skip=${skip}`)
      const j = await r.json(); setItems(j.items||[]); setTotal(j.total||0)
    }finally{ setLoading(false) }
  }
  useEffect(()=>{ load() },[source,page,pageSize])
  return (
    <div>
      <Table
        size="small"
        rowKey="id"
        loading={loading}
        dataSource={items}
        columns={[
          {title:'Chunk ID', dataIndex:'id', render:(v)=> <Typography.Text code>{v}</Typography.Text>},
          {title:'预览', dataIndex:'preview'},
          {title:'操作', render:(_,r)=> <Button size="small" onClick={()=>onOpen(r.id)}>详情</Button>}
        ]}
        pagination={false}
      />
      <div style={{display:'flex', justifyContent:'flex-end', marginTop:8}}>
        <Pagination size="small" current={page} pageSize={pageSize} total={total} onChange={(p,ps)=>{setPage(p); setPageSize(ps)}} showSizeChanger/>
      </div>
    </div>
  )
}

function GraphView({data, visible}:{data:{nodes:any[], edges:any[]}, visible?: boolean}){
  const elements = useMemo(()=>{
    const safeNodes = Array.isArray(data.nodes)? data.nodes:[]
    const safeEdges = Array.isArray(data.edges)? data.edges:[]
    const ns = safeNodes.map((n:any)=>({ data: { id: String(n.id), label: String(n.label||n.id||''), type: n.type || 'chunk' } }))
    const es = safeEdges.map((e:any,i:number)=>({ data: { id: `e${i}`, source: String(e.source), target: String(e.target), label: String(e.type||'REL') } }))
    return [...ns, ...es]
  },[data])
  const layout = { name: 'cose', animate: true }
  const stylesheet = [
    { selector: 'node', style: { 'label': 'data(label)', 'font-size': 10, 'text-valign': 'center', 'background-color': '#5b8ff9' } },
    { selector: 'node[type="entity"]', style: { 'background-color': '#5ad8a6' } },
    { selector: 'edge', style: { 'curve-style': 'bezier', 'line-color': '#ccc', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#ccc', 'width': 1, 'label':'data(label)', 'font-size':8 } }
  ]
  // 处理 Modal 初始隐藏导致的尺寸为 0 的问题：在可见或元素变化时，强制 resize/fit/layout
  const cyRef = useRef<any>(null)
  function onCy(cy:any){
    cyRef.current = cy
    // 初次挂载后稍作延迟，等待容器完成渲染
    setTimeout(()=>{
      try{ cy.resize(); cy.fit(); cy.layout(layout as any).run() }catch{}
    }, 30)
  }
  useEffect(()=>{
    if(!cyRef.current) return
    try{
      cyRef.current.resize()
      cyRef.current.fit()
      cyRef.current.layout(layout as any).run()
    }catch{}
  }, [visible, elements.length])
  try{
    // 空数据占位
    if(elements.length === 0){
      return <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'center', color:'#bbb'}}>暂无邻域</div>
    }
    return (
      <CytoscapeComponent
        key={`graph-${elements.length}`}
        elements={elements as any}
        style={{ width:'100%', height:'100%' }}
        layout={layout as any}
        stylesheet={stylesheet as any}
        cy={onCy as any}
      />)
  }catch{
    return <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'center', color:'#999'}}>图谱渲染失败</div>
  }
}
