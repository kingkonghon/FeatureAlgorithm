import pandas as pd
import numpy as np
from talib import SMA, EMA, WMA, BBANDS, MACD, OBV, AD, ADX, CCI, RSI, STOCH, WILLR

def priceTechnicalIndicator(value, tot_lags, prefix):
    lags = np.array(tot_lags)
    lags = lags[lags >= 2]

    data_result = pd.DataFrame([])
    # MA series
    for lag in lags:
        # SMA
        sma = pd.Series(SMA(value, timeperiod=lag), name='%sMA_%dD' % (prefix, lag))
        data_result = pd.concat([data_result, sma], axis=1)

        # SMA derivatives
        tmp = ma_derivative_indicator(value, sma, 'MA', lag, prefix)
        data_result = data_result.join(tmp)

        # EMA
        ema = pd.Series(EMA(value, lag), name='%sEMA_%dD' % (prefix, lag))
        data_result = data_result.join(ema)

        # EMA derivatives
        tmp = ma_derivative_indicator(value, ema, 'EMA', lag, prefix)
        data_result = data_result.join(tmp)

        # WMA
        wma = pd.Series(WMA(value, lag), name='%sWMA_%dD' % (prefix, lag))
        data_result = data_result.join(wma)

        # WMA derivatives
        tmp = ma_derivative_indicator(value, wma, 'WMA', lag, prefix)
        data_result = data_result.join(tmp)

    # change percent
    lags = tot_lags
    for lag in lags:
        tmp = pd.Series(value.diff(lag) / value.shift(lag), name='%sRET_%dD' % (prefix, lag))
        data_result = data_result.join(tmp)

    # volatility
    lags = np.array(tot_lags)
    lags = lags[lags >= 5]
    for lag in lags:
        tmp = pd.Series(value.rolling(window=lag).std(), name='%sSTD_%dD' % (prefix, lag))
        data_result = data_result.join(tmp)

    # technical indicators
    lags = [7, 14, 21, 28]  # ****** 待修改，技术指标的参数只用最常用的一套
    for lag in lags:
        # bollinger brands
        tmp_upper, tmp_middle, tmp_lower = BBANDS(value, lag, 2, 2)
        tmp_upper.name = '%sBBANDS_UPPER_%dD' % (prefix, lag)
        tmp_lower.name = '%sBBANDS_LOWER_%dD' % (prefix, lag)
        data_result = data_result.join(tmp_upper)
        data_result = data_result.join(tmp_lower)

        # MACD
        tmp, tmp_signal, tmp_hist = MACD(value, 12, 26, lag)
        tmp.name = '%sMACD_DIF_12_26D' % prefix  # macd 对应 macd_dif 线
        tmp_signal.name = '%sMACD_DEA_12_26_%dD' % (prefix, lag)  # macd_signal 对应 macd_dea 线
        tmp_hist = 2 * tmp_hist
        tmp_hist.name = '%sMACD_12_26_%dD' % (prefix, lag)  # 2 * macd_hist 对应 macd 线
        if tmp.name not in data_result.columns:  # macd_dif is the same for all lags
            data_result = data_result.join(tmp)
        data_result = data_result.join(tmp_signal)
        data_result = data_result.join(tmp_hist)


    return data_result

def priceTechnicalIndicatorOHLCV(open_price, high_price, low_price, close_price, volume):
    data_result = pd.DataFrame([])

    lags = [7, 14, 21, 28]  # ****** 待修改，技术指标的参数只用最常用的一套
    # accumlation/distribution
    ad = AD(high_price, low_price, close_price, volume)
    for lag in lags:
        # n day diff of AD
        tmp = ad.diff(lag)
        tmp.name = 'AD_DIFF_%dD' % lag
        data_result = pd.concat([data_result, tmp], axis=1)

        #Average Directional Movement Index
        tmp = ADX(high_price, low_price, close_price, lag)
        tmp.name = 'ADX_%dD' % lag
        data_result = data_result.join(tmp)

        # Commodity Channel Index
        tmp = CCI(high_price, low_price, close_price, lag)
        tmp.name = 'CCI_%dD' % lag
        data_result = data_result.join(tmp)

        # RSI
        tmp = RSI(close_price, lag)
        tmp.name = 'RSI_%dD' % lag
        data_result = data_result.join(tmp)

        # Stochastic
        tmp_k, tmp_d = STOCH(high_price, low_price, close_price, fastk_period=lag, slowk_period=3, slowd_period=3)
        tmp_k.name = 'STOCH_K_%dD' % lag
        tmp_d.name = 'STOCH_D_%dD' % lag
        data_result = data_result.join(tmp_k)
        data_result =data_result.join(tmp_d)

        # WILLR - Williams' %R
        tmp = WILLR(high_price, low_price, close_price, lag)
        tmp.name = 'WILLER_%dD' % lag
        data_result =data_result.join(tmp)

        # volatility ratio
        tmp = VR(high_price, low_price, close_price, lag)
        tmp.name = 'VR_%dD' % lag
        data_result = data_result.join(tmp)

    return data_result

def priceTechnicalIndicatorRollingSum(open_price, high_price, low_price, close_price, volume, turnover, lags):
    # corresponding to dongliangzhibiao
    data_result = pd.DataFrame([])

    for lag in lags:
        # turnover rolling sum
        turnover_rolling_sum = turnover.rolling(window=lag).sum()
        turnover_rolling_sum.name = 'TURNOVER_ROLLING_SUM_%dD' % lag
        data_result = pd.concat([data_result, turnover_rolling_sum], axis=1)

        # Amplitude = (rolling_high - roling_low) / pre close
        pre_close_price = close_price.shift(lag)
        rolling_high = high_price.rolling(window=lag).max()
        rolling_low = low_price.rolling(window=lag).min()
        amplitude = (rolling_high - rolling_low) / pre_close_price
        amplitude.name = 'AMPLITUDE_%dD' % lag
        data_result = data_result.join(amplitude)

    return data_result

def priceOtherIndicatorRanking(values_df, trade_dates, date_field, code_field, data_fields, category_field):
    # no category to divide stocks
    if category_field == 'all':
        data_result = pd.DataFrame([])

        for date in trade_dates:
            tmp_data = values_df.loc[values_df[date_field] == date]
            tmp_rank = tmp_data[data_fields].rank(axis=0, method='average', ascending=False)
            tmp_size = (~tmp_rank.isnull()).sum(axis=0)
            tmp_rank = tmp_rank / tmp_size.astype('float')
            tmp_result = tmp_data[[date_field, code_field]].join(tmp_rank)

            data_result = data_result.append(tmp_result)

        # rename columns
        rename_dict = {}
        for field in data_fields:
            rename_dict[field] = field + '_RANK'
        rename_dict[date_field] = date_field
        rename_dict[code_field] = code_field

        data_result = data_result.rename(columns=rename_dict)

    #  there is category to divide stocks
    else:
        data_result = values_df.copy()

        final_col_names = [date_field, code_field]
        for tmp_field in data_fields:
            tmp_new_field = tmp_field + '_RANK'
            final_col_names.append(tmp_new_field)  # add to the buffer
            # get rank
            data_result.loc[:, tmp_new_field] = data_result.groupby([date_field, category_field])[tmp_field].transform(
                lambda x: x.rank(axis=0, method='average', ascending=False))
            # get max rank
            data_result.loc[:, 'category_size'] = data_result.groupby([date_field, category_field])[tmp_new_field].transform(
                lambda x: x.max())
            # get rank percentage
            data_result.loc[:, tmp_new_field] = data_result[tmp_new_field] / data_result['category_size']

        # drop unnecessary columns
        data_result = data_result[final_col_names]

    return data_result


def volumeTechnicalIndicators(price, volume, tot_lags, prefix):
    data_result = priceTechnicalIndicator(volume, tot_lags, prefix)

    # OBV
    lags = tot_lags
    obv = OBV(price, volume)
    for lag in lags:
        tmp = obv.diff(lag) # diff, otherwise the scale would change if accumulates
        tmp.name = 'OBV_DIFF_%dD' % lag
        data_result = data_result.join(tmp)

    # VOLUME RATIO
    lag = 5
    tmp = volume.rolling(window=lag).mean()
    volume_ratio = volume / tmp
    volume_ratio.name = 'VOLUME_RATIO'
    data_result = data_result.join(volume_ratio)

    return data_result


def ma_derivative_indicator(value, ma, ma_name, lag, prefix):
    data_ret = pd.DataFrame([])

    # value - MA
    tmp = value - ma
    tmp.name = '%s%s_DIFF_%dD' % (prefix, ma_name, lag)
    data_ret = pd.concat([data_ret, tmp], axis=1)

    # sign(value - MA)
    data_ret['%s%s_DIFF_SIGN_%dD' % (prefix, ma_name, lag)] = np.where \
        (data_ret['%s%s_DIFF_%dD' % (prefix, ma_name, lag)] > 0, 1, 0)
    data_ret.loc[np.isnan(data_ret['%s%s_DIFF_%dD' % (prefix, ma_name ,lag)]), '%s%s_DIFF_SIGN_%dD' %
    (prefix, ma_name, lag)] = np.nan

    # count( sign(value - MA) )
    data_ret['%s%s_DIFF_PLUS_COUNT_%dD' % (prefix, ma_name, lag)] = data_ret[
        '%s%s_DIFF_SIGN_%dD' % (prefix, ma_name, lag)].rolling(window=lag).apply(np.count_nonzero)
    data_ret['%s%s_DIFF_NEGATIVE_COUNT_%dD' % (prefix, ma_name, lag)] = lag - data_ret[
        '%s%s_DIFF_PLUS_COUNT_%dD' % (prefix, ma_name, lag)]

    # continuous count( sign(value - MA) )
    # tmp = data_ret['%s%s_DIFF_SIGN_%dD' % (prefix, ma_name, lag)].tolist()
    # continue_up_count = listcomputeincre(tmp)
    # continue_down_count = listcomputeshrink(tmp)
    tmp_up_sign = (data_ret['%s%s_DIFF_SIGN_%dD' % (prefix, ma_name, lag)] == 1).tolist()
    continue_up_count = countContinuousDayNum(tmp_up_sign)
    tmp_down_sign = (data_ret['%s%s_DIFF_SIGN_%dD' % (prefix, ma_name, lag)] == 1).tolist()
    continue_down_count = countContinuousDayNum(tmp_down_sign)

    data_ret['%s%s_DIFF_CON_PLUS_COUNT_%dD' % (prefix, ma_name, lag)] = continue_up_count
    data_ret['%s%s_DIFF_CON_NEGATIVE_COUNT_%dD' % (prefix, ma_name, lag)] = continue_down_count

    return data_ret


def listcomputeincre(lio):
    li = list(reversed(lio))
    rl = []
    i = 0
    j = 0
    k = 0
    while (1):
        if i == len(li):
            break
        elif j == len(li):
            i = i + 1
            j = i
            continue
        elif li[j] == 1:
            j = j + 1
            k = k + 1
        elif np.isnan(li[i]):
            rl.append(li[i])
            i = i + 1
            j = i
        else:
            rl.append(k)
            k = 0
            i = i + 1
            j = i
    return list(reversed(rl))

def countContinuousDayNum(sign):
    count = np.zeros(len(sign))
    for i in range(1, len(sign)):
        if sign[i]:
            count[i] = count[i-1] + 1
        else:
            count[i] = 0

    return count

def listcomputeshrink(lio):
    li = list(reversed(lio))
    rl = []
    i = 0
    j = 0
    k = 0
    while (1):
        if i == len(li):
            break
        elif j == len(li):
            i = i + 1
            j = i
            continue
        elif li[j] == 0:
            j = j + 1
            k = k + 1
        elif np.isnan(li[i]):
            rl.append(li[i])
            i = i + 1
            j = i
        else:
            rl.append(k)
            k = 0
            i = i + 1
            j = i
    return list(reversed(rl))


def VR(highPrice, lowPrice, closePrice, lag):
    high_price = highPrice.copy()
    low_price = lowPrice.copy()
    close_price = closePrice.copy()

    pre_close_price = close_price.shift(1)

    high_price.name = 'high'
    low_price.name = 'low'
    close_price.name = 'close'
    pre_close_price.name = 'pre_close'

    current_high = pd.concat([high_price, pre_close_price], axis=1).max(axis=1, skipna=True)
    current_low = pd.concat([low_price, pre_close_price], axis=1).min(axis=1, skipna=True)
    current_range = current_high - current_low

    n_day_high = current_high.rolling(window=lag, min_periods=1).max()
    n_day_low = current_low.rolling(window=lag, min_periods=1).min()
    n_day_range = n_day_high - n_day_low
    VR = current_range / n_day_range

    return VR

def calWeightedSumIndexQuote(data, quote_fields, date_field, category_field, weight_field):
    basic_data = data.copy()
    # sum total weight (transformation)
    basic_data['tot_weight'] = basic_data.groupby([date_field, category_field])[weight_field].transform(np.sum)
    # calculate real weight (sum to 1)
    basic_data.loc[:, weight_field] = basic_data[weight_field] / basic_data['tot_weight']
    # calculate weighted value
    weighted_fields = list(map(lambda x: 'w_' + x, quote_fields))
    for field in zip(weighted_fields, quote_fields):
        basic_data[field[0]] = basic_data[field[1]] * basic_data[weight_field]
    # sum total weighted value (combine)
    index_quote = basic_data.groupby([date_field, category_field])[weighted_fields].sum()
    index_quote = index_quote.reset_index()

    rename_dict = {}
    for field in zip(weighted_fields, quote_fields):
        rename_dict[field[0]] = field[1]
    index_quote = index_quote.rename(columns=rename_dict)

    return index_quote