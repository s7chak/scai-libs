import pandas as pd
import Config
import yfinance as yf
class Stock_Rater:
    def __init__(self, symbols):
        self.symbols = symbols
        self.tickers = []
        for s in symbols:
            stock = yf.Ticker(s)
            self.tickers.append(stock)
        self.scores = self.score_stocks()

    def score_stocks(self):
        fund_df = pd.DataFrame()
        for ticker in self.tickers:
            t = pd.DataFrame(ticker.info)[:1]
            fund_df = pd.concat([fund_df, t])
        data = fund_df[Config.analysis_fields]
        uvs = self.uvs_scoring(data)
        ufs = self.ufs_scoring(uvs)
        final = self.all_scoring(ufs)
        final['ThresholdScore'] = round((final['UVS'] + final['UFS']) / 20, 2)
        final.sort_values('10Score', ascending=False)
        res = final.T.to_dict('dict')
        return res

    # Fundamental Score
    def ufs_scoring(self, d):
        # UFS = (normalized revenue growth score + normalized earnings growth score + normalized profit margin score + normalized ROE score + normalized debt to equity ratio score + normalized PEG ratio score) / 6
        # w1 = 20, w2 = 20, w3 = 10, w4 = 20, w5 = 15, w6 = 15
        vars = Config.ufs_needs + Config.uvs_needs + [Config.eveb]
        d = d[vars + ['UVS']]
        my_thresholds = Config.ufs_thresholds
        for m in my_thresholds.keys():
            d[m + '_score'] = round((d[m].values[0] - my_thresholds[m][0]) / (my_thresholds[m][1] - my_thresholds[m][0]), 2)
            d[m + '_score'].fillna(0, inplace=True)

        d['UFS'] = round((Config.ufs_weights[Config.rg] * d[Config.rg + '_score'] + Config.ufs_weights[Config.eg] * d[Config.eg + '_score'] + Config.ufs_weights[Config.pm] * d[Config.pm + '_score'] + Config.ufs_weights[
                              Config.roe] * d[Config.roe + '_score'] + Config.ufs_weights[Config.dte] * d[Config.dte + '_score'] + Config.ufs_weights[Config.peg] * d[Config.peg + '_score']) / 100, 2)
        return d.sort_values('UFS', ascending=False)

    def all_scoring(self, d):
        # UFS = (normalized revenue growth score + normalized earnings growth score + normalized profit margin score + normalized ROE score + normalized debt to equity ratio score + normalized PEG ratio score) / 6
        # w1 = 20, w2 = 20, w3 = 10, w4 = 20, w5 = 15, w6 = 15
        vars = Config.ufs_needs + Config.uvs_needs + [Config.eveb, 'UVS', 'UFS']
        d = d[vars]
        d['10Score'] = 0
        my_thresholds = Config.ufs_thresholds
        my_thresholds.update(Config.uvs_thresholds)
        d = d.fillna(0)
        for m in my_thresholds.keys():
            d[m + '_norm'] = round((d[m] - d[m].min()) / (d[m].max() - d[m].min()), 2)
            d['10Score'] += d[m + '_norm']
        d['10Score'] = round(d['10Score']*10/11 , 2)
        d['Missing'] = 0
        return d.sort_values('10Score', ascending=False)

    # Value Score
    def uvs_scoring(self, d):
        # UVS = w1 * (normalized P / E) + w2 * (normalized P / B) + w3 * (normalized P / S) + w4 * (normalized EV / EBITDA) + w5 * (normalized Fwd P / E)
        # w1 = 20, w2 = 20, w3 = 10, w4 = 20, w5 = 30
        vars = Config.uvs_needs + Config.ufs_needs
        d = d[['symbol'] + vars]
        d.set_index('symbol', inplace=True)
        d[Config.eveb] = d[Config.ev] / d[Config.ebitda]
        my_thresholds = Config.uvs_thresholds
        for m in my_thresholds.keys():
            d[m + '_score'] = round((d[m].values[0] - my_thresholds[m][0]) / (my_thresholds[m][1] - my_thresholds[m][0]), 2)
            d[m + '_score'].fillna(0, inplace=True)
        d['UVS'] = round((Config.uvs_weights[Config.pe] * d[Config.pe + '_score'] + Config.uvs_weights[Config.pb] * d[Config.pb + '_score'] + Config.uvs_weights[Config.ps] * d[Config.ps + '_score'] +
                          Config.uvs_weights[Config.eveb] * d[Config.eveb + '_score'] + Config.uvs_weights[Config.fpe] * d[Config.fpe + '_score']) / 100, 2)
        return d.sort_values('UVS', ascending=False)

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    stock_rater = Stock_Rater(tickers)
    print(stock_rater.scores)