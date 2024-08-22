analysis_fields = ['symbol', 'sector', 'currentPrice', 'debtToEquity', 'earningsGrowth', 'ebitda',
                           'enterpriseToEbitda', 'enterpriseValue', 'forwardEps', 'forwardPE', 'payoutRatio',
                           'pegRatio', 'priceToBook', 'priceToSalesTrailing12Months', 'profitMargins', 'quickRatio',
                           'returnOnEquity', 'revenueGrowth', 'targetMedianPrice', 'trailingPE', 'trailingPegRatio']
pe = 'trailingPE'
fpe = 'forwardPE'
ps = 'priceToSalesTrailing12Months'
pb = 'priceToBook'
ev = 'enterpriseValue'
ebitda = 'ebitda'
eveb = 'ev/ebitda'
uvs_needs = [pe,pb,ps,ev,ebitda,fpe]
uvs_weights = {pe:10, pb: 10, ps: 10, eveb:30, fpe: 40}
uvs_thresholds = {pe: [10,20], pb: [1,5], ps: [0.5,3], eveb: [1,10], fpe: [15,70]}

rg = 'revenueGrowth'
eg = 'earningsGrowth'
pm = 'profitMargins'
roe = 'returnOnEquity'
dte = 'debtToEquity'
peg = 'pegRatio'
ufs_needs = [rg, eg, pm, roe, dte, peg]
ufs_weights = {rg: 20, eg: 20, pm: 10, roe: 20, dte: 15, peg:15}
ufs_thresholds = {rg: [5, 20], eg: [5, 20], pm: [10, 30], roe: [10, 30], dte: [0.5, 2], peg: [0.5,2]}