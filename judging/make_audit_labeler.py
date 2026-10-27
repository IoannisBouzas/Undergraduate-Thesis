"""Build a blind, offline labeling page from results/judge_audit_sheet.csv.

Only the sheet (question, gold, model answer) is embedded; the key is never read, so the
page cannot leak judge verdicts. Labels autosave in the browser and export as a CSV in the
sheet's own format, ready for: python -m judging.build_audit_sheet --score <exported csv>
"""
import csv, json
from pathlib import Path

SHEET = Path("results/judge_audit_sheet.csv")
OUT = Path("results/judge_audit_labeler.html")
LABEL_COL = "human_label(correct/incorrect/ambiguous)"

rows = [{"id": r["id"], "q": r["question"], "gold": r["gold_answer"], "pred": r["model_answer"]}
        for r in csv.DictReader(SHEET.open(encoding="utf-8-sig"))]
assert len(rows) == 100 and all(set(r) == {"id", "q", "gold", "pred"} for r in rows)

html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Judge Audit</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--bd:#e7e5e4;--c:#15803d;--i:#b91c1c;--a:#a16207}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--bd:#44403c}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0;padding:16px}
main{max-width:820px;margin:auto}.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:18px;margin:12px 0}
.k{color:var(--mut);font-size:13px;text-transform:uppercase;letter-spacing:.04em}.v{white-space:pre-wrap;margin:2px 0 14px}
button{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid var(--bd);background:var(--card);color:var(--fg);cursor:pointer}
.sel-correct{background:var(--c);color:#fff}.sel-incorrect{background:var(--i);color:#fff}.sel-ambiguous{background:var(--a);color:#fff}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.bar{height:6px;background:var(--bd);border-radius:3px}
.bar>div{height:6px;background:var(--c);border-radius:3px}details{color:var(--mut);font-size:14px}
</style></head><body><main>
<h1 style="font-size:20px;margin:0">Blind judge audit</h1>
<details><summary>Rubric (same standard the judge was asked to apply)</summary>
<p><b>Correct</b>: the model answer contains the gold answer or is equivalent to it (paraphrase, synonym,
same date in a different format). Extra correct detail is fine. A partial answer that misses a required part is incorrect.
<b>Gold is a refusal</b> (for example "not enough information"): correct only if the model also declines.
<b>Ambiguous</b>: only when the gold itself is unclear or you genuinely cannot decide. Judge the answer, not the gold;
if you think the gold is wrong, label against the gold and use Ambiguous sparingly.</p>
<p>Keys: <b>C</b> correct, <b>I</b> incorrect, <b>A</b> ambiguous, <b>&larr;/&rarr;</b> move, <b>N</b> next unlabelled.
Do not open results/judge_audit_key.csv until you have exported.</p></details>
<div class="card"><div class="row" style="justify-content:space-between"><span id="pos"></span><span id="done"></span></div>
<div class="bar"><div id="prog"></div></div></div>
<div class="card"><div class="k">Question</div><div class="v" id="q"></div>
<div class="k">Gold answer</div><div class="v" id="g"></div>
<div class="k">Model answer</div><div class="v" id="p"></div>
<div class="row"><button data-l="correct">Correct (C)</button><button data-l="incorrect">Incorrect (I)</button>
<button data-l="ambiguous">Ambiguous (A)</button></div></div>
<div class="row"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
<button id="nextU">Next unlabelled (N)</button><button id="exp" style="margin-left:auto">Export CSV</button></div>
<p class="k" id="save"></p>
</main><script>
const ROWS=__ROWS__;const KEY="judge_audit_labels_v1";
let L={};try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(L));$("save").textContent="saved in this browser"}catch(e){$("save").textContent="browser storage unavailable: export before closing"}}
function show(){const r=ROWS[i];$("q").textContent=r.q;$("g").textContent=r.gold;$("p").textContent=r.pred||"(empty answer)";
$("pos").textContent="Item "+r.id+" of "+ROWS.length;const n=Object.keys(L).length;$("done").textContent=n+" labelled";
$("prog").style.width=(n/ROWS.length*100)+"%";document.querySelectorAll("[data-l]").forEach(b=>b.className=L[r.id]===b.dataset.l?"sel-"+b.dataset.l:"")}
function lab(l){L[ROWS[i].id]=l;save();if(i<ROWS.length-1)i++;show()}
function nextU(){for(let k=1;k<=ROWS.length;k++){const j=(i+k)%ROWS.length;if(!L[ROWS[j].id]){i=j;return show()}}show()}
document.querySelectorAll("[data-l]").forEach(b=>b.onclick=()=>lab(b.dataset.l));
$("prev").onclick=()=>{if(i>0)i--;show()};$("next").onclick=()=>{if(i<ROWS.length-1)i++;show()};$("nextU").onclick=nextU;
document.onkeydown=e=>{const k=e.key.toLowerCase();if(k==="c")lab("correct");else if(k==="i")lab("incorrect");else if(k==="a")lab("ambiguous");
else if(k==="arrowleft")$("prev").click();else if(k==="arrowright")$("next").click();else if(k==="n")nextU()};
$("exp").onclick=()=>{const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
const lines=[["id","question","gold_answer","model_answer","__LABEL__"].map(q).join(",")];
ROWS.forEach(r=>lines.push([r.id,r.q,r.gold,r.pred,L[r.id]||""].map(q).join(",")));
const blob=new Blob(["\\ufeff"+lines.join("\\r\\n")],{type:"text/csv"});const a=document.createElement("a");
a.href=URL.createObjectURL(blob);a.download="judge_audit_sheet_filled.csv";a.click()};
const first=ROWS.findIndex(r=>!L[r.id]);i=first<0?0:first;show();
</script></body></html>"""

OUT.write_text(html.replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
               .replace("__LABEL__", LABEL_COL), encoding="utf-8")
print(f"wrote {OUT} ({len(rows)} items, no key embedded)")
