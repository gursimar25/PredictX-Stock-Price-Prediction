let candleChart = null, modelChart = null;
const $ = id => document.getElementById(id);
const money = v => Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});

async function analyze(){
 const ticker=$('ticker').value.trim(), period=$('period').value, model=$('model').value;
 if(!ticker){$('message').textContent='Please enter a stock symbol.';return;}
 $('message').textContent='Fetching Yahoo Finance data and training models…';
 try{
  const [r,m]=await Promise.all([
   fetch(`/api/stock?ticker=${encodeURIComponent(ticker)}&period=${period}`),
   fetch(`/api/models?ticker=${encodeURIComponent(ticker)}&period=${period}`)
  ]);
  const d=await r.json(), md=await m.json();
  if(!d.success) throw Error(d.error); if(!md.success) throw Error(md.error);
  const selected=md.models[model] || md.models['Linear Regression'];
  $('latest').textContent=money(d.latest); $('predicted').textContent=money(selected.next_price); $('difference').textContent=(selected.next_price-d.latest>=0?'+':'')+money(selected.next_price-d.latest);
  $('symbol').textContent=d.ticker; $('modelLabel').textContent=model+' · educational estimate'; $('mae').textContent=money(selected.mae); $('rmse').textContent=money(selected.rmse); $('r2').textContent=Number(selected.r2).toFixed(2);
  const ind=d.indicators; const last=a=>{for(let i=a.length-1;i>=0;i--)if(a[i]!=null)return a[i];return null};
  $('rsi').textContent=last(ind.RSI14)?.toFixed(2)||'—'; $('macd').textContent=last(ind.MACD)?.toFixed(2)||'—'; $('sma').textContent=last(ind.SMA20)?money(last(ind.SMA20)):'—'; $('ema').textContent=last(ind.EMA20)?money(last(ind.EMA20)):'—';
  renderCandles(d); renderModels(md.models);
  $('message').textContent=`Loaded ${d.ticker} · ${d.dates.length} sessions · 80/20 chronological validation.`;
 }catch(e){console.error(e);$('message').textContent=e.message;}
}
function renderCandles(d){
 const el=$('candleChart'); el.innerHTML=''; if(candleChart)candleChart.remove();
 candleChart=LightweightCharts.createChart(el,{layout:{background:{color:'transparent'},textColor:'#7b879b'},grid:{vertLines:{color:'#18202c'},horzLines:{color:'#18202c'}},rightPriceScale:{borderColor:'#283346'},timeScale:{borderColor:'#283346',timeVisible:false}});
 const candle=candleChart.addCandlestickSeries({upColor:'#58c68d',downColor:'#ef7180',borderVisible:false,wickUpColor:'#58c68d',wickDownColor:'#ef7180'});
 candle.setData(d.dates.map((date,i)=>({time:date,open:d.open[i],high:d.high[i],low:d.low[i],close:d.close[i]})));
 const add=(arr)=>candleChart.addLineSeries({lineWidth:1,lastValueVisible:false,priceLineVisible:false}).setData(d.dates.map((date,i)=>({time:date,value:arr[i]})).filter(x=>x.value!=null));
 add(d.indicators.SMA20); add(d.indicators.EMA20); add(d.indicators.BBUpper); add(d.indicators.BBLower); candleChart.timeScale().fitContent();
}
function renderModels(models){
 const body=$('modelRows'); body.innerHTML=''; const labels=[],vals=[];
 Object.entries(models).forEach(([name,r])=>{const tr=document.createElement('tr'); [name,money(r.next_price),money(r.mae),money(r.rmse),Number(r.r2).toFixed(3)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});body.appendChild(tr);labels.push(name);vals.push(r.next_price);});
 if(modelChart)modelChart.destroy(); modelChart=new Chart($('modelChart'),{type:'bar',data:{labels,datasets:[{label:'Next-price estimate',data:vals,borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#69758a'}},y:{ticks:{color:'#69758a'}}}}});
}
async function savePrediction(){
 const token=document.querySelector('meta[name="csrf-token"]')?.content; const ticker=$('ticker').value.trim(),period=$('period').value,model=$('model').value;
 if(!token){$('message').textContent='Security token not found. Refresh the page.';return;}
 try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json','X-CSRFToken':token},body:JSON.stringify({ticker,period,model})});const d=await r.json();if(!r.ok||!d.success)throw Error(d.error||'Unable to save prediction.');$('message').textContent=`Saved ${d.ticker} prediction using ${d.model}: ${money(d.predicted)}`;}catch(e){$('message').textContent=e.message;}
}
async function addWatchlist(){try{const ticker=$('ticker').value.trim();const r=await fetch('/api/watchlist',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker})});const d=await r.json();$('message').textContent=d.success?'Added '+ticker.toUpperCase()+' to watchlist.':d.error;}catch(e){$('message').textContent=e.message;}}
async function runLSTM(){const ticker=$('ticker').value.trim(),period=$('period').value;$('lstmResult').textContent='Training LSTM… this can take a little while.';try{const r=await fetch(`/api/lstm?ticker=${encodeURIComponent(ticker)}&period=${period}`);const d=await r.json();if(!d.success)throw Error(d.error);$('lstmResult').textContent=`LSTM estimate: ${money(d.next_price)} · MAE ${money(d.mae)} · RMSE ${money(d.rmse)} · R² ${Number(d.r2).toFixed(3)}`;}catch(e){$('lstmResult').textContent=e.message;}}
const tickerInput=$('ticker'), suggestionsBox=$('stockSuggestions'); let searchTimeout;
if(tickerInput&&suggestionsBox){tickerInput.addEventListener('input',function(){const q=this.value.trim();clearTimeout(searchTimeout);if(!q){suggestionsBox.innerHTML='';suggestionsBox.style.display='none';return;}searchTimeout=setTimeout(async()=>{try{const r=await fetch(`/api/search?q=${encodeURIComponent(q)}`),stocks=await r.json();suggestionsBox.innerHTML='';if(!Array.isArray(stocks)||!stocks.length){suggestionsBox.style.display='none';return;}stocks.forEach(stock=>{const item=document.createElement('div');item.className='stock-suggestion';const s=document.createElement('div'),n=document.createElement('div');s.className='stock-symbol';n.className='stock-name';s.textContent=stock.symbol;n.textContent=stock.name;item.append(s,n);item.onclick=()=>{tickerInput.value=stock.symbol;suggestionsBox.style.display='none';};suggestionsBox.appendChild(item)});suggestionsBox.style.display='block';}catch(e){suggestionsBox.style.display='none';}},300)});document.addEventListener('click',e=>{if(!e.target.closest('.stock-search'))suggestionsBox.style.display='none';});}
window.addEventListener('DOMContentLoaded',()=>{const q=new URLSearchParams(location.search).get('ticker');if(q&&$('ticker'))$('ticker').value=q;analyze();});
// ============================================================
// PREDICTX AI ANALYST
// ============================================================
// ============================================================
// PREDICTX AI ANALYST - FREE VERSION
// ============================================================

async function askAI() {

    const tickerInput = document.getElementById("ticker");
    const aiText = document.getElementById("aiText");

    if (!tickerInput || !aiText) {

        console.error("AI elements not found.");

        return;
    }

    const ticker = tickerInput.value.trim().toUpperCase();

    if (!ticker) {

        aiText.innerText =
            "Please enter a stock symbol first.";

        return;
    }

    // --------------------------------------------------------
    // Read current PredictX results
    // --------------------------------------------------------

    const latestElement =
        document.getElementById("latest");

    const predictedElement =
        document.getElementById("predicted");

    const differenceElement =
        document.getElementById("difference");

    const maeElement =
        document.getElementById("mae");

    const rmseElement =
        document.getElementById("rmse");

    const r2Element =
        document.getElementById("r2");

    const modelElement =
        document.getElementById("model");

    const rsiElement =
        document.getElementById("rsi");

    const macdElement =
        document.getElementById("macd");

    const smaElement =
        document.getElementById("sma");

    const emaElement =
        document.getElementById("ema");


    // --------------------------------------------------------
    // Get displayed values
    // --------------------------------------------------------

    const latest =
        latestElement?.innerText || "0";

    const predicted =
        predictedElement?.innerText || "0";

    const difference =
        differenceElement?.innerText || "0";

    const mae =
        maeElement?.innerText || "0";

    const rmse =
        rmseElement?.innerText || "0";

    const r2 =
        r2Element?.innerText || "0";

    const model =
        modelElement?.value || "Machine Learning model";

    const rsi =
        rsiElement?.innerText || "N/A";

    const macd =
        macdElement?.innerText || "N/A";

    const sma =
        smaElement?.innerText || "N/A";

    const ema =
        emaElement?.innerText || "N/A";


    // --------------------------------------------------------
    // Show loading message
    // --------------------------------------------------------

    aiText.innerHTML =
        "🤖 <strong>PredictX AI Analyst</strong><br><br>" +
        "Analyzing the ML prediction, model performance " +
        "and technical indicators...";


    try {

        // ----------------------------------------------------
        // Get CSRF token
        // ----------------------------------------------------

        const csrfToken =
            document
                .querySelector('meta[name="csrf-token"]')
                ?.getAttribute("content");


        // ----------------------------------------------------
        // Send data to Flask
        // ----------------------------------------------------

        const response =
            await fetch(
                "/api/ai-analysis",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "X-CSRFToken":
                            csrfToken
                    },

                    body: JSON.stringify({

                        ticker: ticker,

                        latest: latest,

                        predicted: predicted,

                        difference: difference,

                        mae: mae,

                        rmse: rmse,

                        r2: r2,

                        model: model,

                        rsi: rsi,

                        macd: macd,

                        sma: sma,

                        ema: ema
                    })
                }
            );


        const data =
            await response.json();


        // ----------------------------------------------------
        // Handle server error
        // ----------------------------------------------------

        if (!response.ok) {

            aiText.innerText =
                data.error ||
                "AI analysis failed.";

            return;
        }


        // ----------------------------------------------------
        // Display analysis
        // ----------------------------------------------------

        aiText.innerText =
            data.analysis ||
            "No analysis was generated.";


    } catch (error) {

        console.error(
            "PredictX AI Analyst Error:",
            error
        );

        aiText.innerText =
            "Unable to generate the AI analysis. " +
            "Please try again.";

    }

}
