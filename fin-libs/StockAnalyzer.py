import pandas as pd
import Config
import yfinance as yf
class Stock_Rater:
    def __init__(self, symbols):
        self.symbols = symbols
        self.tickers = []
        self.stock_data = None

    def load_stocks_info(self):
        fund_df = pd.DataFrame()
        for s in self.symbols:
            stock = yf.Ticker(s)
            self.tickers.append(stock)
        for ticker in self.tickers:
            try:
                t = pd.DataFrame(ticker.info)[:1]
                fund_df = pd.concat([fund_df, t])
            except:
                print(str(ticker))

        data = fund_df[Config.analysis_fields]
        data.set_index('symbol', inplace=True)
        self.stock_data = data

    def enrich_stocks(self):
        if self.stock_data is None or self.stock_data.empty:
            self.load_stocks_info()
        if not any(['Score' in x for x in self.stock_data.columns]):
            self.scores = self.score_stocks()
            self.stock_data = self.stock_data.merge(self.scores, left_index=True, right_index=True, how='left')
        return self.stock_data

    def score_stocks(self):
        if self.stock_data is None or self.stock_data.empty:
            self.load_stocks_info()
        # uvs = self.uvs_scoring(data)
        # ufs = self.ufs_scoring(uvs)
        # final = self.all_scoring(ufs)
        # final['ThresholdScore'] = round((final['UVS'] + final['UFS']) / 20, 2)
        # final.sort_values('10Score', ascending=False)
        # res = final.T.to_dict('dict')
        # return res
        data = self.stock_data
        past_scores = self.past_score(data)
        future_scores = self.future_score(data)
        final = past_scores.merge(future_scores, left_index=True, right_index=True)
        final['OverallScore'] = (final['PastScore'] + final['FutureScore']) / 2
        final['SellScore'] = final['PastScore'] - final['FutureScore']
        final['MissingScore'] = (final['Missing_p'] + final['Missing_f']) / 2
        final.sort_values('OverallScore', ascending=False, inplace=True)
        final.fillna(0, inplace=True)
        return final

    def normalize(self, data, factor):
        """Helper function to normalize data between 0 and 1."""
        min_val = data[factor].min()
        max_val = data[factor].max()
        if max_val == min_val:  # To avoid division by zero
            return pd.Series(0.5, index=data.index)  # Assign neutral score if there's no variation
        return (data[factor] - min_val) / (max_val - min_val)

    def past_score(self, data):
        # Normalize and calculate the Past Score
        past_scores = pd.DataFrame(index=data.index)
        missing_counts = pd.Series(0, index=data.index)
        for factor in Config.past_score_needs:
            if factor in data.columns:
                past_scores[factor + '_norm'] = self.normalize(data, factor)
                missing_counts += data[factor].isna().astype(int)
            else:
                past_scores[factor + '_norm'] = 0
                missing_counts += 1
        past_scores['PastScore'] = round(past_scores.mean(axis=1) * 10)
        past_scores['Missing_p'] = (missing_counts / len(Config.past_score_needs)) * 100
        past_scores = past_scores[~past_scores.index.duplicated(keep='first')]
        return past_scores[['PastScore', 'Missing_p']]

    def future_score(self, data):
        # Normalize and calculate the Future Score
        future_scores = pd.DataFrame(index=data.index)
        missing_counts = pd.Series(0, index=data.index)
        for factor in Config.future_score_needs:
            if factor in data.columns:
                future_scores[factor + '_norm'] = self.normalize(data, factor)
                missing_counts += data[factor].isna().astype(int)
            else:
                future_scores[factor + '_norm'] = 0
                missing_counts += 1

        future_scores['FutureScore'] = round(future_scores.mean(axis=1) * 10)
        future_scores['Missing_f'] = (missing_counts / len(Config.past_score_needs)) * 100
        future_scores = future_scores[~future_scores.index.duplicated(keep='first')]
        return future_scores[['FutureScore', 'Missing_f']]


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
        # MinMax Scaled score of all ufs, uvs factors
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

# if __name__ == "__main__":
#     tickers = ['AAPL', 'AMZN','AMC','ABCB','BW','BAC','GOOGL', 'MSFT','SCHW','JPM','JNJ','RDDT','DTC','PARA','PFE','TOYOF', 'TM','INTC']
#     stock_rater = Stock_Rater(tickers)
#     df = stock_rater.enrich_stocks()
#     print(df.head())