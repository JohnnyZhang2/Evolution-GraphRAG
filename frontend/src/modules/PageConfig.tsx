import { useEffect, useMemo, useState } from 'react'
import { Button, Checkbox, Space, message, Form, Input, InputNumber, Switch, Collapse, Divider, Tooltip, Typography, Select } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'
const { Panel } = Collapse
const { Text } = Typography
const API = import.meta.env.VITE_API_BASE || ''

type FieldMeta = {
  key: string
  label: string
  type: 'string'|'number'|'boolean'|'password'
  desc?: string
  min?: number
  max?: number
  step?: number
}

const SECTIONS: Array<{title:string, fields: FieldMeta[]}> = [
  {
    title: 'Neo4j 连接',
    fields: [
      { key: 'neo4j_uri', label: 'URI', type: 'string', desc: '数据库连接字符串：本地常用 bolt://localhost:7687；Neo4j Aura 使用 neo4j+s://…' },
      { key: 'neo4j_user', label: '用户名', type: 'string', desc: '数据库用户名（默认为 neo4j 或自定义用户）' },
      { key: 'neo4j_password', label: '密码', type: 'password', desc: '数据库密码（出于安全不会回显，留空则不修改）' },
    ]
  },
  {
    title: 'LLM / Embedding',
    fields: [
      { key: 'llm_base_url', label: 'LLM Base URL', type: 'string', desc: 'OpenAI 兼容服务地址，例如 http://127.0.0.1:1234' },
      { key: 'llm_api_key', label: 'LLM API Key', type: 'string', desc: '本地 LM Studio 通常可留空' },
      { key: 'llm_model', label: 'LLM 模型', type: 'string', desc: '回答所用模型名称' },
      { key: 'embedding_model', label: 'Embedding 模型', type: 'string', desc: '向量检索所用 Embedding 模型名称' },
      { key: 'embedding_batch_size', label: 'Embed 批大小', type: 'number', desc: 'Embedding 请求的批处理条数', min:1, max:512 },
      { key: 'embedding_timeout', label: 'Embed 超时(秒)', type: 'number', desc: '单次 Embedding 请求超时时间', min:1, max:600 },
    ]
  },
  {
    title: '检索与扩展',
    fields: [
      { key: 'top_k', label: 'Top K', type: 'number', desc: '初始向量检索召回数量', min:1, max:200 },
      { key: 'expand_hops', label: '扩展跳数', type: 'number', desc: '上下文扩展的跳数（>1 将纳入关系/共现等）', min:1, max:3 },
      { key: 'bm25_enabled', label: '启用 BM25', type: 'boolean', desc: '混合检索：在向量得分基础上加入 BM25 文本检索' },
      { key: 'bm25_weight', label: 'BM25 权重', type: 'number', desc: 'BM25 对最终得分的占比系数（0-1）', min:0, max:1, step:0.05 },
      { key: 'graph_rank_enabled', label: '启用图中心性加成', type: 'boolean', desc: '按图节点度/中心性为候选加分' },
      { key: 'graph_rank_weight', label: '图加成权重', type: 'number', desc: '中心性加成的强度（0-1）', min:0, max:1, step:0.05 },
    ]
  },
  {
    title: '重排序（Rerank）',
    fields: [
      { key: 'rerank_enabled', label: '启用 Rerank', type: 'boolean', desc: '对初筛候选做交叉编码重排序' },
      { key: 'rerank_model', label: 'Rerank 模型', type: 'string', desc: '例如 bge-reranker-large；留空使用降级策略' },
      { key: 'rerank_endpoint', label: 'Rerank 接口', type: 'string', desc: 'HTTP POST 接口地址（自建/托管服务）' },
      { key: 'rerank_top_n', label: 'Rerank Top N', type: 'number', desc: '进入 heavy rerank 的最大候选数', min:1, max:500 },
      { key: 'rerank_timeout', label: 'Rerank 超时(秒)', type: 'number', desc: '单次 rerank 请求超时', min:1, max:120 },
      { key: 'rerank_alpha', label: '融合系数 alpha', type: 'number', desc: 'final = alpha*composite + (1-alpha)*rerank', min:0, max:1, step:0.05 },
      { key: 'rerank_cb_fails', label: '熔断失败阈值', type: 'number', desc: '连续失败次数触发熔断', min:0, max:10 },
      { key: 'rerank_cb_cooldown', label: '熔断冷却(秒)', type: 'number', desc: '熔断后冷却时间', min:0, max:600 },
      { key: 'rerank_cache_ttl', label: '结果缓存 TTL(秒)', type: 'number', desc: 'rerank 结果的缓存时间', min:0, max:600 },
    ]
  },
  {
    title: '子图提取（Subgraph）',
    fields: [
      { key: 'subgraph_enable', label: '启用子图扩展', type: 'boolean', desc: '基于初始命中构建实体/关系邻域' },
      { key: 'subgraph_max_nodes', label: '子图最大节点', type: 'number', desc: '扩展节点上限', min:10, max:1000 },
      { key: 'subgraph_max_depth', label: '最大深度', type: 'number', desc: '逻辑深度（跳数）', min:1, max:5 },
      { key: 'subgraph_rel_types', label: '关系类型限定', type: 'string', desc: "逗号分隔或 '*' 表示全部" },
      { key: 'subgraph_weight', label: '子图加权', type: 'number', desc: '子图节点额外加权', min:0, max:1, step:0.05 },
      { key: 'subgraph_per_entity_limit', label: '每实体上限', type: 'number', desc: '单个实体引出的最大 chunk 数', min:0, max:20 },
      { key: 'subgraph_depth_decay', label: '深度衰减', type: 'number', desc: 'depth>=2 时加成衰减', min:0, max:1, step:0.05 },
      { key: 'subgraph_rel_multiplier', label: '关系节点乘数', type: 'number', desc: '关系型子图节点额外乘数', min:0, max:5, step:0.1 },
      { key: 'subgraph_rel_min_keep', label: '最少关系节点', type: 'number', desc: '保留关系节点的最少数量', min:0, max:50 },
      { key: 'subgraph_rel_multi_scale', label: '多关系缩放', type: 'number', desc: '对其他关系累积加成的缩放', min:0, max:2, step:0.05 },
      { key: 'subgraph_rel_hits_decay', label: '多关系统计衰减', type: 'number', desc: '按排名的衰减系数', min:0, max:1, step:0.05 },
      { key: 'subgraph_rel_hits_max', label: '多关系最大数', type: 'number', desc: '参与多关系加权的最大关系条数', min:0, max:20 },
      { key: 'subgraph_rel_density_cap', label: '密度惩罚阈值', type: 'number', desc: '超过该密度开始惩罚', min:0, max:50 },
      { key: 'subgraph_rel_density_alpha', label: '密度惩罚强度', type: 'number', desc: '惩罚强度系数', min:0, max:1, step:0.05 },
      { key: 'subgraph_depth1_cap', label: '深度1新增上限', type: 'number', desc: 'depth=1 允许新增的节点数', min:0, max:200 },
      { key: 'subgraph_depth1_rel_cap', label: '深度1关系新增上限', type: 'number', desc: '通过关系新增的最大数量', min:0, max:200 },
      { key: 'subgraph_deep_reserve_nodes', label: '深层预留配额', type: 'number', desc: '为更深层预留节点数', min:0, max:200 },
      { key: 'subgraph_path_score_enable', label: '路径评分', type: 'boolean', desc: '对子图节点计算 path_score' },
      { key: 'subgraph_path_entity_base', label: '实体路径基础值', type: 'number', desc: '实体共享路径基础值', min:0, max:2, step:0.05 },
      { key: 'subgraph_path_entity_decay', label: '实体路径深度衰减', type: 'number', desc: '实体路径深度衰减', min:0, max:1, step:0.05 },
      { key: 'subgraph_path_rel_conf_weight', label: '关系路径置信权重', type: 'number', desc: 'conf * weight 系数', min:0, max:2, step:0.05 },
      { key: 'subgraph_path_rel_decay', label: '关系路径深度衰减', type: 'number', desc: '关系路径深度衰减', min:0, max:1, step:0.05 },
      { key: 'subgraph_path_score_weight', label: '路径分数权重', type: 'number', desc: 'path_score 融合系数', min:0, max:2, step:0.05 },
      { key: 'subgraph_path_max_records', label: '路径记录上限', type: 'number', desc: '每节点保留的最大路径条目', min:0, max:50 },
      { key: 'id_normalize_enable', label: 'ID 规范化', type: 'boolean', desc: '对子图匹配的 ID 做规范化' },
    ]
  },
  {
    title: '实体/关系 Schema',
    fields: [
      { key: 'entity_typed_mode', label: '实体带类型', type: 'boolean', desc: '实体抽取返回 {name,type}' },
      { key: 'entity_types', label: '实体类型白名单', type: 'string', desc: '逗号分隔类型列表；用于 typed 模式' },
      { key: 'relation_enforce_types', label: '强制关系白名单', type: 'boolean', desc: '只保留白名单内的关系类型' },
      { key: 'relation_types', label: '关系类型白名单', type: 'string', desc: '逗号分隔类型列表' },
      { key: 'relation_fallback_type', label: '非法类型回退', type: 'string', desc: '强制白名单时非法类型回退为该类型（留空则丢弃）' },
      { key: 'entity_min_length', label: '实体最小长度', type: 'number', desc: '短实体清理阈值', min:1, max:10 },
      { key: 'cooccur_min_count', label: '共现最小计数', type: 'number', desc: '共现边低于该计数则清理', min:1, max:10 },
    ]
  },
  {
    title: '关系抽取',
    fields: [
      { key: 'relation_extraction', label: '启用关系抽取', type: 'boolean', desc: '在导入时调用 LLM 抽取 Chunk 间语义关系' },
      { key: 'relation_window', label: '关系窗口', type: 'number', desc: '相邻窗口大小：前后各 N 个 chunk 组合', min:1, max:8 },
      { key: 'relation_chunk_trunc', label: '关系抽取截断', type: 'number', desc: '传给 LLM 的单个 Chunk 最大字符数', min:100, max:2000, step:50 },
      { key: 'relation_llm_temperature', label: '关系温度', type: 'number', desc: '关系抽取时 LLM 的温度', min:0, max:1, step:0.05 },
      { key: 'relation_min_confidence', label: '最小置信度', type: 'number', desc: '在扩展阶段保留的关系最低置信度阈值', min:0, max:1, step:0.05 },
    ]
  },
  {
    title: '上下文裁剪与其他',
    fields: [
      { key: 'context_max', label: '上下文上限', type: 'number', desc: '最终送入 LLM 的最大上下文条数', min:1, max:200 },
      { key: 'context_min_per_reason', label: '来源最低保留', type: 'number', desc: '每类来源（向量/实体/关系/共现/LLM 关系）最少保留条数', min:0, max:10 },
      { key: 'entity_normalize_enabled', label: '实体标准化', type: 'boolean', desc: '启用同义词合并与规范化' },
      { key: 'synonyms_file', label: '同义词文件', type: 'string', desc: 'JSON 或 TSV 文件路径（alt→canonical）' },
    ]
  }
]

export function PageConfig(){
  const [form] = Form.useForm()
  const [persist, setPersist] = useState(false)
  const [diag, setDiag] = useState<any>(null)
  const [raw, setRaw] = useState<any>({})
  const [loading, setLoading] = useState(false)
  const defaultEntityTypes = ['Person','Organization','Location','Product','Concept','Event']
  const defaultRelTypes = ['STEP_NEXT','CAUSES','SUPPORTS','REFERENCES','PART_OF','SUBSTEP_OF','CONTRASTS','FOLLOWS']

  async function load(){
    setLoading(true)
    try{
      const r = await fetch(`${API}/config`)
      const j = await r.json();
      setRaw(j)
      // 填充表单；password 不回显
      const initial: any = {}
      for(const sec of SECTIONS){
        for(const f of sec.fields){
          if(f.key === 'neo4j_password') continue
          if(j.hasOwnProperty(f.key)) initial[f.key] = j[f.key]
        }
      }
      // 列表类字段转换为数组供多选控件使用
      const toArr = (s?:string) => (s? s.split(',').map(x=>x.trim()).filter(Boolean) : [])
      initial.entity_types = toArr(j.entity_types)
      initial.relation_types = toArr(j.relation_types)
      initial.subgraph_rel_types = (j.subgraph_rel_types === '*' ? ['*'] : toArr(j.subgraph_rel_types))
      form.setFieldsValue(initial)
    }catch(e:any){
      message.error(`读取配置失败：${e?.message||e}`)
    }finally{
      setLoading(false)
    }
  }

  function buildPayload(values:any){
    const payload:any = {}
    for(const sec of SECTIONS){
      for(const f of sec.fields){
        if(f.key === 'neo4j_password'){
          const val = values[f.key]
          if(val && String(val).length>0) payload[f.key] = val
          continue
        }
        if(values.hasOwnProperty(f.key)) payload[f.key] = values[f.key]
      }
    }
    // 多选控件回写为逗号分隔
    const toCsv = (arr:any) => (Array.isArray(arr)? arr.filter(Boolean).join(',') : undefined)
    if(Array.isArray(values.entity_types)) payload.entity_types = toCsv(values.entity_types)
    if(Array.isArray(values.relation_types)) payload.relation_types = toCsv(values.relation_types)
    if(Array.isArray(values.subgraph_rel_types)) payload.subgraph_rel_types = (values.subgraph_rel_types.includes('*')? '*' : toCsv(values.subgraph_rel_types))
    // 空字符串回退为 null
    if(typeof values.relation_fallback_type === 'string' && values.relation_fallback_type.trim()==='') payload.relation_fallback_type = null
    return payload
  }

  async function save(){
    try{
      const values = await form.validateFields()
      const payload = buildPayload(values)
      const url = `${API}/config${persist ? `?persist=true` : ''}`
      const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})
      if(r.ok){
        const j = await r.json()
        message.success(`已更新：${(j.updated||[]).join(', ') || '无'}${j.persisted ? '（已写入 .env）' : ''}`)
        // 保存后刷新一次原始配置（便于摘要对齐）
        load()
      } else {
        const err = await r.json().catch(()=>({detail:r.statusText}))
        message.error(`保存失败：${err.detail || r.statusText}`)
      }
    }catch(e:any){
      if(e?.errorFields){
        message.error('请检查表单项')
      }else{
        message.error(`保存失败：${e?.message||e}`)
      }
    }
  }

  async function test(){
    try{
      const r = await fetch(`${API}/diagnostics`)
      const j = await r.json(); setDiag(j)
      const ok = j?.neo4j?.neo4j === true
      message[ok? 'success':'warning'](`Neo4j: ${ok? 'OK':'异常'}`)
    }catch(e:any){
      message.error(`诊断失败：${e?.message||e}`)
    }
  }

  function getRules(key:string){
    if(key==='neo4j_uri'){
      return [{
        validator: (_:any, val:any)=>{
          if(!val) return Promise.resolve()
          const ok = /^(bolt|neo4j|neo4j\+s):\/\//i.test(String(val))
          return ok? Promise.resolve() : Promise.reject(new Error('需以 bolt:// 或 neo4j:// 或 neo4j+s:// 开头'))
        }
      }]
    }
    if(key==='llm_base_url' || key==='rerank_endpoint'){
      return [{
        validator: (_:any, val:any)=>{
          if(!val) return Promise.resolve()
          try{
            const u = new URL(String(val))
            if(u.protocol==='http:'||u.protocol==='https:') return Promise.resolve()
            return Promise.reject(new Error('仅支持 http/https'))
          }catch{ return Promise.reject(new Error('URL 格式不正确')) }
        }
      }]
    }
    return []
  }

  useEffect(()=>{ load() },[])

  const advancedJson = useMemo(()=> JSON.stringify(raw, null, 2), [raw])

  return (
    <div className="card">
      <h3>配置</h3>
      <div className="row" style={{marginBottom:8}}>
        <Space>
          <Button onClick={load} loading={loading}>读取</Button>
          <Button type="primary" onClick={save} loading={loading}>保存</Button>
          <Checkbox checked={persist} onChange={e=>setPersist(e.target.checked)}>写入 .env</Checkbox>
          <Button onClick={test}>测试连接</Button>
        </Space>
      </div>
      <Form form={form} layout="vertical">
        {SECTIONS.map((sec)=> (
          <div key={sec.title} style={{border:'1px solid #eee', borderRadius:8, padding:12, marginBottom:12}}>
            <Divider orientation="left" style={{marginTop:0}}>{sec.title}</Divider>
            <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(260px,1fr))', gap:12}}>
              {sec.fields.map(f=> (
                <Form.Item key={f.key} name={f.key} rules={getRules(f.key)} label={<>
                  {f.label} {f.desc && <Tooltip title={f.desc}><InfoCircleOutlined style={{opacity:0.6}}/></Tooltip>}
                </>}>
                  {f.type==='string' && (
                    f.key==='relation_types' ? (
                      <Select mode="tags" placeholder="选择或输入关系类型" options={defaultRelTypes.map(v=>({label:v, value:v}))} />
                    ) : f.key==='entity_types' ? (
                      <Select mode="tags" placeholder="选择或输入实体类型" options={defaultEntityTypes.map(v=>({label:v, value:v}))} />
                    ) : f.key==='subgraph_rel_types' ? (
                      <Select mode="multiple" placeholder="选择关系类型或 * 全部" allowClear options={[{label:'* (全部)', value:'*'}, ...defaultRelTypes.map(v=>({label:v, value:v}))]} />
                    ) : f.key==='relation_fallback_type' ? (
                      <Select allowClear placeholder="留空则丢弃非法类型" options={defaultRelTypes.map(v=>({label:v, value:v}))} />
                    ) : (
                      <Input placeholder={`请输入 ${f.label}`} />
                    )
                  )}
                  {f.type==='password' && <Input.Password placeholder='留空则不修改' visibilityToggle />}
                  {f.type==='number' && <InputNumber style={{width:'100%'}} min={f.min} max={f.max} step={f.step} />}
                  {f.type==='boolean' && <Switch />}
                </Form.Item>
              ))}
            </div>
          </div>
        ))}
      </Form>

      <Collapse style={{marginTop:8}}>
        <Panel header={<span>高级（原始 JSON 只读概览）</span>} key="adv">
          <Text type="secondary">完整配置快照（只读）。如需调整未暴露的字段，可在上方保存后再通过 .env 手工编辑。</Text>
          <pre style={{marginTop:8, background:'#f6f6f6', padding:8, maxHeight:300, overflow:'auto'}}>{advancedJson}</pre>
        </Panel>
        {diag && (
          <Panel header={<span>诊断结果</span>} key="diag">
            <pre style={{marginTop:8, background:'#f6f6f6', padding:8, maxHeight:300, overflow:'auto'}}>{JSON.stringify(diag, null, 2)}</pre>
          </Panel>
        )}
      </Collapse>
    </div>
  )
}
