import { useEffect, useState } from 'react'
import { Table, Modal, Typography, Pagination } from 'antd'
const API = import.meta.env.VITE_API_BASE || ''

export function PageLogs(){
  const [items, setItems] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [loading, setLoading] = useState(false)
  const [detail, setDetail] = useState<{open:boolean, data?:any}>({open:false})

  async function load(){
    setLoading(true)
    try{
      const skip = (page-1)*pageSize
      const r = await fetch(`${API}/qa/logs?limit=${pageSize}&skip=${skip}`)
      const j = await r.json(); setItems(j.items||[]); setTotal(j.total||0)
    }finally{ setLoading(false) }
  }
  useEffect(()=>{ load() },[page,pageSize])
  return (
    <div className="card">
      <h3>问答日志</h3>
      <Table
        rowKey="id"
        dataSource={items}
        loading={loading}
        pagination={false}
        onRow={r=>({ onClick: ()=> setDetail({open:true, data:r}) })}
        columns={[
          {title:'时间', dataIndex:'updatedAt', width:200, render:(v)=> v ? new Date(v).toLocaleString() : ''},
          {title:'问题', dataIndex:'question'},
          {title:'回答摘要', dataIndex:'answer_preview'},
        ]}
      />
      <div style={{display:'flex', justifyContent:'flex-end', marginTop:8}}>
        <Pagination current={page} pageSize={pageSize} total={total} onChange={(p,ps)=>{setPage(p); setPageSize(ps)}} showSizeChanger/>
      </div>
      <Modal open={detail.open} onCancel={()=>setDetail({open:false})} footer={null} title="问答详情" width={900}>
        {detail.data && (
          <div>
            <Typography.Paragraph><strong>问题：</strong>{detail.data.question}</Typography.Paragraph>
            <Typography.Paragraph><strong>回答：</strong>{detail.data.answer || ''}</Typography.Paragraph>
          </div>
        )}
      </Modal>
    </div>
  )
}
