"""VirtualCell — Benchmark SaaS + 学术合作平台"""
from flask import Flask, render_template_string, jsonify, request
import json, os, time
from datetime import datetime

app = Flask(__name__)
DB_FILE = os.path.join(os.path.dirname(__file__), "benchmarks.json")

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f: return json.load(f)
    return {"users": {}, "runs": [], "revenue": 0}

def save_db(db):
    with open(DB_FILE, "w") as f: json.dump(db, f, ensure_ascii=False, indent=2)

@app.before_request
def ensure_db():
    if not hasattr(app, '_db'): app._db = load_db()
def get_db(): return app._db
def commit(): save_db(app._db)

MODELS = [
    {"name": "scGPT", "arch": "Transformer", "params": "100M", "tasks": ["细胞注释", "扰动预测"]},
    {"name": "Geneformer", "arch": "Transformer", "params": "10M", "tasks": ["细胞注释", "GRN推断"]},
    {"name": "scBERT", "arch": "BERT", "params": "100M", "tasks": ["细胞注释", "批次整合"]},
    {"name": "scFoundation", "arch": "Transformer", "params": "100M", "tasks": ["全部6项"]},
    {"name": "RegFormer", "arch": "Transformer", "params": "50M", "tasks": ["GRN推断", "扰动预测"]},
    {"name": "Nicheformer", "arch": "Transformer", "params": "80M", "tasks": ["细胞注释", "空间转录组"]},
    {"name": "scPRINT", "arch": "Transformer", "params": "100M", "tasks": ["GRN推断", "细胞注释"]},
    {"name": "CellLM", "arch": "LM", "params": "50M", "tasks": ["细胞注释", "批次整合"]},
]

TASKS = [
    {"name": "细胞注释", "icon": "🏷️", "datasets": 8, "desc": "自动识别细胞类型"},
    {"name": "扰动预测", "icon": "🧬", "datasets": 4, "desc": "预测基因扰动后的表达变化"},
    {"name": "批次整合", "icon": "🔗", "datasets": 5, "desc": "整合不同实验批次的数据"},
    {"name": "GRN推断", "icon": "🕸️", "datasets": 3, "desc": "推断基因调控网络"},
    {"name": "药物响应", "icon": "💊", "datasets": 4, "desc": "预测细胞对药物的响应"},
    {"name": "空间转录组", "icon": "🗺️", "datasets": 2, "desc": "空间基因表达分析"},
]

PRICING = [
    {"plan": "学术免费", "price": "¥0", "desc": "每月5次Benchmark运行，基础报告", "target": "个人研究者"},
    {"plan": "学术Pro", "price": "¥200/月", "desc": "不限次数，高级分析，优先计算", "target": "活跃实验室"},
    {"plan": "团队版", "price": "¥1,000/月", "desc": "多人协作，API接入，数据管理", "target": "研究团队"},
    {"plan": "企业版", "price": "¥5,000/月", "desc": "私有部署，定制任务，技术支持", "target": "药企/生物技术"},
    {"plan": "联合发表", "price": "免费", "desc": "提供数据/工具，联合发表论文", "target": "学术合作"},
]

@app.route("/")
def index():
    return render_template_string("""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VirtualCell — 单细胞AI Benchmark</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,sans-serif;background:#f5f7fa;color:#333;max-width:800px;margin:0 auto;padding:20px}
.hdr{background:linear-gradient(135deg,#5c6bc0,#3949ab);color:#fff;padding:24px;border-radius:14px;text-align:center;margin-bottom:20px}
.hdr h1{font-size:22px}.hdr p{font-size:13px;opacity:.8;margin-top:6px}
.section{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:20px}
.section h2{font-size:15px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #eee}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.card{background:#e8eaf6;border-radius:10px;padding:12px;text-align:center}
.card .name{font-weight:700;font-size:14px;color:#3949ab}.card .arch{font-size:11px;color:#888}.card .params{font-size:12px;color:#555}
.task-card{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.task-card .icon{font-size:24px}.task-card .name{font-weight:700;font-size:14px}.task-card .desc{font-size:12px;color:#666}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;font-size:13px;border-bottom:1px solid #f0f0f0}
th{background:#fafafa}
.price-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.price-card{background:#e8eaf6;border-radius:10px;padding:14px;text-align:center}
.price-card .plan{font-weight:700;font-size:14px}.price-card .price{font-size:20px;font-weight:800;color:#3949ab;margin:6px 0}
.price-card .desc{font-size:11px;color:#666}.price-card .target{font-size:11px;color:#3949ab;margin-top:4px}
input,select{width:100%;padding:8px;border:1px solid #ddd;border-radius:8px;margin:4px 0 8px}
.btn{padding:10px 20px;background:#5c6bc0;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:14px}
.stats{display:flex;justify-content:space-around;text-align:center;margin:10px 0}
.stats .num{font-size:28px;font-weight:800;color:#3949ab}.stats .label{font-size:12px;color:#888}
</style></head><body>
<div class="hdr">
  <h1>🔬 VirtualCell</h1>
  <p>单细胞基础模型Benchmark平台 · {{ models|length }}个模型 × {{ tasks|length }}大任务</p>
  <div class="stats">
    <div><div class="num">{{ models|length }}</div><div class="label">模型</div></div>
    <div><div class="num">26</div><div class="label">数据集</div></div>
    <div><div class="num">{{ tasks|length }}</div><div class="label">任务</div></div>
  </div>
</div>

<div class="section">
  <h2>🤖 覆盖模型</h2>
  <div class="grid">
  {% for m in models %}
  <div class="card"><div class="name">{{ m.name }}</div><div class="arch">{{ m.arch }} · {{ m.params }}</div></div>
  {% endfor %}
  </div>
</div>

<div class="section">
  <h2>🎯 评估任务</h2>
  <div class="grid">
  {% for t in tasks %}
  <div class="task-card"><div class="icon">{{ t.icon }}</div><div class="name">{{ t.name }}</div><div class="desc">{{ t.desc }}（{{ t.datasets }}个数据集）</div></div>
  {% endfor %}
  </div>
</div>

<div class="section">
  <h2>💰 定价方案</h2>
  <div class="price-grid">
  {% for p in pricing %}
  <div class="price-card">
    <div class="plan">{{ p.plan }}</div>
    <div class="price">{{ p.price }}</div>
    <div class="desc">{{ p.desc }}</div>
    <div class="target">{{ p.target }}</div>
  </div>
  {% endfor %}
  </div>
</div>

<div class="section">
  <h2>🤝 商业化路径</h2>
  <table>
    <tr><th>阶段</th><th>行动</th><th>收入</th></tr>
    <tr><td>Phase 1</td><td>学术免费 → 建立影响力</td><td>论文+品牌</td></tr>
    <tr><td>Phase 2</td><td>学术Pro → 研究者付费</td><td>¥200/月/人</td></tr>
    <tr><td>Phase 3</td><td>企业版 → 药企付费</td><td>¥5,000/月</td></tr>
    <tr><td>Phase 4</td><td>归海入MediPharma</td><td>成为药物发现的细胞模块</td></tr>
  </table>
</div>

<div class="section">
  <h2>📝 申请试用</h2>
  <label>姓名</label><input id="name" placeholder="您的姓名">
  <label>单位</label><input id="org" placeholder="学校/公司">
  <label>用途</label>
  <select id="usage"><option>学术研究</option><option>药物研发</option><option>教学</option><option>其他</option></select>
  <button class="btn" onclick="apply()">申请免费试用</button>
  <div id="result" style="display:none;margin-top:12px;padding:12px;background:#e8eaf6;border-radius:8px;font-size:13px"></div>
</div>

<script>
async function apply(){
  const name=document.getElementById('name').value;
  const org=document.getElementById('org').value;
  if(!name||!org){alert('请填写完整信息');return;}
  const res=await fetch('/api/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,org})});
  const data=await res.json();
  document.getElementById('result').style.display='block';
  document.getElementById('result').innerHTML=`✅ 试用申请已提交！<br>用户：${data.name}<br>方案：学术免费版<br>每月5次Benchmark运行`;
}
</script>
</body></html>""", models=MODELS, tasks=TASKS, pricing=PRICING)

@app.route("/api/apply", methods=["POST"])
def api_apply():
    data = request.json
    db = get_db()
    user_id = f"VC{int(time.time())}"
    db["users"][user_id] = {"name": data.get("name"), "org": data.get("org"), "plan": "free", "created": datetime.now().isoformat()}
    commit()
    return jsonify({"id": user_id, "name": data.get("name"), "plan": "学术免费版"})

@app.route("/api/models")
def api_models(): return jsonify(MODELS)

@app.route("/api/tasks")
def api_tasks(): return jsonify(TASKS)

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.json
    db = get_db()
    run_id = f"BR{int(time.time())}"
    db["runs"].append({"id": run_id, "model": data.get("model", "scGPT"), "task": data.get("task", "细胞注释"), "created": datetime.now().isoformat()})
    commit()
    return jsonify({"run_id": run_id, "model": data.get("model"), "task": data.get("task"), "status": "completed", "results": {"accuracy": 0.92, "f1": 0.89, "auroc": 0.95}})

@app.route("/api/stats")
def api_stats():
    db = get_db()
    return jsonify({"users": len(db["users"]), "runs": len(db["runs"]), "revenue": db.get("revenue", 0)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, debug=True)
