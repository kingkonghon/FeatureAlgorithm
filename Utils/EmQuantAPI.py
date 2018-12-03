# -*- coding:utf-8 -*-
__author__ = 'weijie'

"""
*   EmQuantAPI for python
*   version  2.3.0.1
*   c++ version 2.3.0.1
*   Copyright (c) EastMoney Corp. All rights reserved.
"""

from ctypes import *
import sys
import os
from datetime import datetime, date, time

eVT_null = 0
eVT_char = 1
eVT_byte = 2
eVT_bool = 3
eVT_short = 4
eVT_ushort = 5
eVT_int = 6
eVT_uInt = 7
eVT_int64 = 8
eVT_uInt64 = 9
eVT_float = 10
eVT_double = 11
eVT_byteArray = 12
eVT_asciiString = 13
eVT_unicodeString = 14

ePT_NONE = 0        #不使用代理
ePT_HTTP = 1	    #HTTP代理
ePT_HTTPS = 2	    #HTTPS代理
ePT_SOCK4 = 3       #SOCK4代理
ePT_SOCK5 = 4       #SOCK5代理

eOT_default = 0      # 默认(默认则根据传入数量的正负标志买入eOT_buy卖出eOT_Sell,其余类型对数量作正负转换)
eOT_buy = 1          # 买入
eOT_sell = 2         # 卖出
eOT_purchase = 3     # 申购
eOT_redemption = 4   # 赎回

class c_safe_union(Union):
    _fields_ = [
        ("charValue", c_char),
        ("boolValue", c_bool),
        ("shortValue", c_short),
        ("uShortValue", c_ushort),
        ("intValue", c_int),
        ("uIntValue", c_uint),
        ("int64Value", c_longlong),
        ("uInt64Value", c_ulonglong),
        ("floatValue", c_float),
        ("doubleValue", c_double)
    ]


class stEQChar(Structure):
    _fields_ = [
        ("pChar", c_char_p),
        ("nSize", c_uint)
    ]


class stEQCharArray(Structure):
    _fields_ = [
        ("pChArray", POINTER(stEQChar)),
        ("nSize", c_uint)
    ]


class stEQVarient(Structure):
    pass


stEQVarient._fields_ = [
    ("vtype", c_int),
    ("unionValues", c_safe_union),
    ("eqchar", stEQChar)
]


class stEQVarientArray(Structure):
    _fields_ = [
        ("pEQVarient", POINTER(stEQVarient)),
        ("nSize", c_uint)
    ]


class stEQData(Structure):
    _fields_ = [
        ("codeArray", stEQCharArray),
        ("indicatorArray", stEQCharArray),
        ("dateArray", stEQCharArray),
        ("valueArray", stEQVarientArray)
    ]


class stEQLoginInfo(Structure):
    _fields_ = [
        #("userName", c_char_p),
        #("password", c_char_p)
        ("userName", c_char * 255),
        ("password", c_char * 255)
    ]


class stEQMessage(Structure):
    _fields_ = [
        ("version", c_int),
        ("msgType", c_int),
        ("err", c_int),
        ("requestID", c_int),
        ("serialID", c_int),
        ("pEQData", POINTER(stEQData))
    ]

class stEQCtrData(Structure):
    _fields_ = [
        ("row", c_int),
        ("column", c_int),
        ("indicatorArray", stEQCharArray),
        ("valueArray", stEQVarientArray)
    ]

class stOrderInfo(Structure):
    _pack_=8
    _fields_ = [
        ("code", c_char*16),
        ("volume", c_double),
        ("price", c_float),
        ("date", c_int),
        ("time", c_int),
        ("optype", c_int),
        ("cost", c_float),
        ("rate", c_float),
        ("reserve", c_int)
    ]

appLogCallback = None
appQuoteCallback = None
appQuoteFunctionDict = {0:{}, 10000:{}, 10001:{}}
c_LogCallback = CFUNCTYPE(c_int, c_char_p)
c_DataCallback = CFUNCTYPE(c_int, POINTER(stEQMessage), c_void_p)
c_encodeType = "gbk"

class c:
    class EmQuantData:
        def __init__(self, NullValue = None):
            self.ErrorCode = 0
            self.ErrorMsg = "success"
            self.Codes = list()
            self.Indicators = list()
            self.Dates = list()
            self.RequestID = 0
            self.SerialID = 0
            self.Data = dict()
            self.__NullValue = NullValue

        def __str__(self):
            return "ErrorCode=%s, ErrorMsg=%s, Data=%s" % (self.ErrorCode, self.ErrorMsg, str(self.Data))

        def __repr__(self):
            return "ErrorCode=%s, ErrorMsg=%s, Data=%s" % (self.ErrorCode, self.ErrorMsg, str(self.Data))

        def resolve2RankData(self, indicatorData, **arga):
            for i in range(0, indicatorData.codeArray.nSize):
                self.Codes.append(indicatorData.codeArray.pChArray[i].pChar.decode(c_encodeType))
            for k in range(0, indicatorData.indicatorArray.nSize):
                self.Indicators.append(indicatorData.indicatorArray.pChArray[k].pChar.decode(c_encodeType))
            for j in range(0, indicatorData.dateArray.nSize):
                self.Dates.append(indicatorData.dateArray.pChArray[j].pChar.decode(c_encodeType))

            self.Data = []

            for i in range(0, len(self.Codes)):
                for j in range(0, len(self.Indicators)):
                    for k in range(0, len(self.Dates)):
                        self.Data.append(self.getIndicatorDataByIndex(i, j, k, indicatorData))

        def resolve25RankData(self, indicatorData, **arga):
            for i in range(0, indicatorData.codeArray.nSize):
                self.Codes.append(indicatorData.codeArray.pChArray[i].pChar.decode(c_encodeType))
            for k in range(0, indicatorData.indicatorArray.nSize):
                self.Indicators.append(indicatorData.indicatorArray.pChArray[k].pChar.decode(c_encodeType))
            for j in range(0, indicatorData.dateArray.nSize):
                self.Dates.append(indicatorData.dateArray.pChArray[j].pChar.decode(c_encodeType))
            for i in range(0, len(self.Codes)):
                stockCode = self.Codes[i]
                self.Data[stockCode] = []
                for j in range(0, len(self.Indicators)):
                    tempData = None
                    for k in range(0, len(self.Dates)):
                        tempData = self.getIndicatorDataByIndex(i, j, k, indicatorData)
                        self.Data[stockCode].append(tempData)


        def resolve26RankData(self, indicatorData, **arga):
            for i in range(0, indicatorData.codeArray.nSize):
                self.Codes.append(indicatorData.codeArray.pChArray[i].pChar.decode(c_encodeType))
            for k in range(0, indicatorData.indicatorArray.nSize):
                self.Indicators.append(indicatorData.indicatorArray.pChArray[k].pChar.decode(c_encodeType))
            for j in range(0, indicatorData.dateArray.nSize):
                self.Dates.append(indicatorData.dateArray.pChArray[j].pChar.decode(c_encodeType))
            self.Data = []
            for i in range(0, len(self.Codes)):
                for j in range(0, len(self.Indicators)):
                    tempData = []
                    for k in range(0, len(self.Dates)):
                        tempData.append(self.getIndicatorDataByIndex(i, j, k, indicatorData))
                    self.Data.append(tempData)


        def resolve3RankData(self, indicatorData, **arga):
            for i in range(0, indicatorData.codeArray.nSize):
                self.Codes.append(indicatorData.codeArray.pChArray[i].pChar.decode(c_encodeType))
            for k in range(0, indicatorData.indicatorArray.nSize):
                self.Indicators.append(indicatorData.indicatorArray.pChArray[k].pChar.decode(c_encodeType))
            for j in range(0, indicatorData.dateArray.nSize):
                self.Dates.append(indicatorData.dateArray.pChArray[j].pChar.decode(c_encodeType))
            for i in range(0, len(self.Codes)):
                stockCode = self.Codes[i]
                self.Data[stockCode] = []
                for j in range(0, len(self.Indicators)):
                    tempData = []
                    for k in range(0, len(self.Dates)):
                        tempData.append(self.getIndicatorDataByIndex(i, j, k, indicatorData))
                    self.Data[stockCode].append(tempData)

        def resolveCtrData(self, indicatorData, **arga):
            for i in range(0, indicatorData.column):
                self.Indicators.append(indicatorData.indicatorArray.pChArray[i].pChar.decode(c_encodeType))
            for r in range(0, indicatorData.row):
                list1 = []
                for n in range(0, indicatorData.column):
                    list1.append(self.resolve(indicatorData.valueArray.pEQVarient[indicatorData.column * r + n]))
                self.Data[str(r)] = list1

        def resolve(self, variant):
            if variant.vtype == eVT_null:
                return self.__NullValue
            elif variant.vtype == eVT_char:
                return variant.unionValues.charValue
            elif variant.vtype == eVT_bool:
                return variant.unionValues.boolValue
            elif variant.vtype == eVT_short:
                return variant.unionValues.shortValue
            elif variant.vtype == eVT_ushort:
                return variant.unionValues.uShortValue
            elif variant.vtype == eVT_int:
                return variant.unionValues.intValue
            elif variant.vtype == eVT_uInt:
                return variant.unionValues.uIntValue
            elif variant.vtype == eVT_int64:
                return variant.unionValues.int64Value
            elif variant.vtype == eVT_uInt64:
                return variant.unionValues.uInt64Value
            elif variant.vtype == eVT_float:
                return round(variant.unionValues.floatValue, 6)
            elif variant.vtype == eVT_double:
                return round(variant.unionValues.doubleValue, 6)
            elif variant.vtype == eVT_asciiString:
                if variant.eqchar.pChar != None:
                    return variant.eqchar.pChar.decode(c_encodeType)
                else:
                    return ""
            elif variant.vtype == eVT_unicodeString:
                if variant.eqchar.pChar != None:
                    return variant.eqchar.pChar.decode(c_encodeType)
                else:
                    return ""
            return self.__NullValue

        def getIndicatorDataByIndex(self, codeIndex, indicatorIndex, dateIndex, indicatorData):
            if indicatorData.valueArray.nSize == 0:
                return self.__NullValue
            codeSize = indicatorData.codeArray.nSize
            indicatorSize = indicatorData.indicatorArray.nSize
            dateSize = indicatorData.dateArray.nSize
            valueSize = indicatorData.valueArray.nSize
            if valueSize != codeSize * dateSize * indicatorSize:
                return self.__NullValue
            if codeIndex <= codeSize * indicatorSize * dateIndex + indicatorSize * codeIndex + indicatorIndex:
                tempIndex = codeSize * indicatorSize * dateIndex + indicatorSize * codeIndex + indicatorIndex
                return self.resolve(indicatorData.valueArray.pEQVarient[tempIndex])

    apiPackagePath = "."


    for x in sys.path:
        xx = x.find("site-packages")
        if(xx >= 0 and x[xx:] == "site-packages"):
            apiPackagePath = x
            break

    apiPackagePath = os.path.join(apiPackagePath, "EmQuantAPI.pth")
    pthFile = open(apiPackagePath, "r")
    baseDir = pthFile.readline()
    pthFile.close()

    libsDir = os.path.join(baseDir, "libs")
    apiDllPath = os.path.join(libsDir, "EmQuantAPI.dll")

    bit =32
    if sys.maxsize > 2**32:
        bit = 64
    if bit == 64:
        apiDllPath = os.path.join(libsDir, "EmQuantAPI_x64.dll")

    quantLib =CDLL(apiDllPath)

    quant_start = quantLib.start
    quant_start.restype = c_int
    quant_start.argtypes = [POINTER(stEQLoginInfo), c_char_p, c_LogCallback]

    quant_stop = quantLib.stop
    quant_stop.restype = c_int
    quant_stop.argtypes = []

    quant_setcallback = quantLib.setcallback
    quant_setcallback.restype = c_int
    quant_setcallback.argtypes = [c_DataCallback]

    quant_geterrstring = quantLib.geterrstring
    quant_geterrstring.restype = c_char_p
    quant_geterrstring.argtypes = [c_int, c_int]

    quant_csd = quantLib.csd
    quant_csd.restype = c_int
    quant_csd.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_void_p]

    quant_css = quantLib.css
    quant_css.restype = c_int
    quant_css.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]

    quant_tradedates = quantLib.tradedates
    quant_tradedates.restype = c_int
    quant_tradedates.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]

    quant_sector = quantLib.sector
    quant_sector.restype = c_int
    quant_sector.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]

    quant_getdate = quantLib.getdate
    quant_getdate.restype = c_int
    quant_getdate.argtypes = [c_char_p, c_int, c_char_p, c_void_p]

    quant_csc = quantLib.csc
    quant_csc.restype = c_int
    quant_csc.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_void_p]

    quant_cmc = quantLib.cmc
    quant_cmc.restype = c_int
    quant_cmc.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_void_p]

    quant_chmc = quantLib.chmc
    quant_chmc.restype = c_int
    quant_chmc.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_void_p]

    quant_releasedata = quantLib.releasedata
    quant_releasedata.restype = c_int
    quant_releasedata.argtypes = [c_void_p]

    quant_csq = quantLib.csq
    quant_csq.restype = c_int
    quant_csq.argtypes = [c_char_p, c_char_p, c_char_p, c_DataCallback, c_void_p]

    quant_csqcancel = quantLib.csqcancel
    quant_csqcancel.restype = c_int
    quant_csqcancel.argtypes = [c_int]

    quant_cst = quantLib.cst
    quant_cst.restype = c_int
    quant_cst.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_DataCallback, c_void_p]

    quant_csqsnapshot = quantLib.csqsnapshot
    quant_csqsnapshot.restype = c_int
    quant_csqsnapshot.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]

    quant_ctr = quantLib.ctr
    quant_ctr.restype = c_int
    quant_ctr.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]

    quant_cps = quantLib.cps
    quant_cps.restype = c_int
    quant_cps.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_void_p]

    quant_setserverlistdir = quantLib.setserverlistdir
    quant_setserverlistdir.restype = c_voidp
    quant_setserverlistdir.argtypes = [c_char_p]

    quant_setproxy = quantLib.setproxy
    quant_setproxy.restype = c_int
    quant_setproxy.argtypes = [c_int, c_char_p, c_ushort, c_bool, c_char_p, c_char_p]

    quant_manualactivate = quantLib.manualactivate
    quant_manualactivate.restype = c_int
    quant_manualactivate.argtypes = [POINTER(stEQLoginInfo), c_char_p, c_LogCallback]

    quant_pquery = quantLib.pquery
    quant_pquery.restype = c_int
    quant_pquery.argtypes = [c_char_p, c_void_p]

    quant_porder = quantLib.porder
    quant_porder.restype = c_int
    quant_porder.argtypes = [POINTER(stOrderInfo), c_int, c_char_p, c_char_p, c_char_p]

    quant_edb = quantLib.edb
    quant_edb.restype = c_int
    quant_edb.argtypes = [c_char_p, c_char_p, c_void_p]

    quant_edbquery = quantLib.edbquery
    quant_edbquery.restype = c_int
    quant_edbquery.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]

    quant_pcreate = quantLib.pcreate
    quant_pcreate.restype = c_int
    quant_pcreate.argtypes = [c_char_p, c_char_p, c_int64, c_char_p, c_char_p]

    quant_pdelete = quantLib.pdelete
    quant_pdelete.restype = c_int
    quant_pdelete.argtypes = [c_char_p, c_char_p]

    quant_preport = quantLib.preport
    quant_preport.restype = c_int
    quant_preport.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]


    @staticmethod
    def start(options="", logcallback=None, mainCallBack=None):
        """
        初始化和登陆(开始时调用)  options：附加参数 "TestLatency=1"
        :param uname: 用户名
        :param password: 密码
        :param options:可选参数
        :param logcallback:启动结果提示回调函数
        :return:
        """
        if(options.upper().find("LANGUAGEVERSION") == -1):
            options = "LANGUAGEVERSION=4,"+options
        data = c.EmQuantData()
        loginInfo = stEQLoginInfo()
        loginInfo.userName = "".encode(c_encodeType)
        loginInfo.password = "".encode(c_encodeType)

        global appQuoteFunctionDict
        appQuoteFunctionDict[0][0] = mainCallBack

        global appLogCallback
        if callable(logcallback):
            appLogCallback = c_LogCallback(logcallback)
        else:
            def log(logMessage):
                print("[EmQuantAPI Python]", logMessage.decode(c_encodeType))
                return 1
            appLogCallback = c_LogCallback(log)
        c.quant_setserverlistdir(c.libsDir.encode(c_encodeType))
        c.quant_setcallback(appQuoteCallback)
        loginResult = c.quant_start(pointer(loginInfo), options.encode(c_encodeType), appLogCallback)

        if loginResult != 0:
            data.ErrorCode = loginResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)

        return data

    @staticmethod
    def stop():
        """
        退出(结束时调用)
        :return: 0-成功
        """
        data = c.EmQuantData()
        data.ErrorCode = c.quant_stop()
        data.ErrorMsg = c.geterrstring(data.ErrorCode)
        return data

    @staticmethod
    def geterrstring(errcode, lang=1):
        """
        获取错误码文本说明
        :param errcode:错误代码
        :param lang:语言类型 0-中文  1-英文
        :return:
        """
        return c.quant_geterrstring(errcode, lang).decode(c_encodeType)

    @staticmethod
    def csd(codes, indicators, startdate=None, enddate=None, options="", *arga, **argb):
        """
        序列数据查询(同步请求)
        :param codes: 东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param startdate:开始日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param enddate:截止日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :return:EmQuantData
        """
        codes = c.__toString(codes)
        indicators = c.__toString(indicators)

        result = c.__resolveOption(options)
        options = result[0]

        ShowBlank = c.__ShowBlankOption(options)
        data = c.EmQuantData(ShowBlank)

        if(enddate == None):
            enddate = datetime.today().strftime("%Y-%m-%d")
        if(startdate == None):
            startdate = enddate
        if(isinstance(startdate, datetime) or isinstance(startdate, date)):
            startdate = startdate.strftime("%Y-%m-%d")
        if(isinstance(enddate, datetime) or isinstance(enddate, date)):
            enddate = enddate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_csd(codes.encode(c_encodeType), indicators.encode(c_encodeType), startdate.encode(c_encodeType), enddate.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve3RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return c.__tryResolvePandas(data, result[1])

    @staticmethod
    def css(codes, indicators, options="", *arga, **argb):
        """
        截面数据查询(同步请求)
        :param codes:东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :return:EmQuantData
        """
        codes = c.__toString(codes)
        indicators = c.__toString(indicators)
        result = c.__resolveOption(options)
        options = result[0]

        ShowBlank = c.__ShowBlankOption(options)

        data = c.EmQuantData(ShowBlank)

        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_css(codes.encode(c_encodeType), indicators.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve25RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return c.__tryResolvePandas(data, result[1])

    @staticmethod
    def tradedates(startdate=None, enddate=None, options=None, *arga, **argb):
        """
        获取区间日期内的交易日(同步请求)
        :param startdate:开始日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param enddate:截止日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :return:EmQuantData
        """

        if options == None:
            options = ""
        data = c.EmQuantData()

        if(enddate == None):
            enddate = datetime.today().strftime("%Y-%m-%d")
        if(startdate == None):
            startdate = enddate
        if(isinstance(startdate, datetime) or isinstance(startdate, date)):
            startdate = startdate.strftime("%Y-%m-%d")
        if(isinstance(enddate, datetime) or isinstance(enddate, date)):
            enddate = enddate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_tradedates(startdate.encode(c_encodeType), enddate.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve2RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def sector(pukeycode, tradedate, options="", *arga, **argb):
        """
        获取系统板块成分(同步请求)
        :param pukeycode:
        :param tradedate:交易日
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :param arga:
        :param argb:
        :return:
        """
        data = c.EmQuantData()

        if(tradedate == None):
            tradedate = datetime.today().strftime("%Y-%m-%d")
        if(isinstance(tradedate, datetime) or isinstance(tradedate, date)):
            tradedate = tradedate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_sector(pukeycode.encode(c_encodeType), tradedate.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve2RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def getdate(tradedate, offday=0, options="", *arga, **argb):
        """
        获取偏移N的交易日(同步请求)
        :param tradedate:交易日期
        :param offday:偏移天数
        :param options:
        :param arga:
        :param argb:
        :return:
        """
        data = c.EmQuantData()
        if(tradedate == None):
            tradedate = datetime.today().strftime("%Y-%m-%d")
        if(isinstance(tradedate, datetime) or isinstance(tradedate, date)):
            tradedate = tradedate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_getdate(tradedate.encode(c_encodeType), offday, options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve2RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def csc(code, indicators, startdate=None, enddate=None, options="", *arga, **argb):
        """
        历史分钟K线(同步请求) //code只支持单个股票
        :param code: 东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param startdate:开始日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param enddate:截止日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :return:EmQuantData
        """
        code = c.__toString(code)
        indicators = c.__toString(indicators)
        result = c.__resolveOption(options)
        options = result[0]
        data = c.EmQuantData()

        if(enddate == None):
            enddate = datetime.today().strftime("%Y-%m-%d")
        if(startdate == None):
            startdate = enddate
        if(isinstance(startdate, datetime) or isinstance(startdate, date)):
            startdate = startdate.strftime("%Y-%m-%d")
        if(isinstance(enddate, datetime) or isinstance(enddate, date)):
            enddate = enddate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_csc(code.encode(c_encodeType), indicators.encode(c_encodeType), startdate.encode(c_encodeType), enddate.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve26RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return c.__tryResolvePandas(data, result[1])

    @staticmethod
    def cmc(code, indicators, startdate=None, enddate=None, options="", *arga, **argb):
        """
        历史分钟K线(同步请求) //code只支持单个股票
        :param code: 东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param startdate:开始日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param enddate:截止日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :return:EmQuantData
        """
        code = c.__toString(code)
        indicators = c.__toString(indicators)
        result = c.__resolveOption(options)
        options = result[0]
        data = c.EmQuantData()

        if(enddate == None):
            enddate = datetime.today().strftime("%Y-%m-%d")
        if(startdate == None):
            startdate = enddate
        if(isinstance(startdate, datetime) or isinstance(startdate, date)):
            startdate = startdate.strftime("%Y-%m-%d")
        if(isinstance(enddate, datetime) or isinstance(enddate, date)):
            enddate = enddate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_cmc(code.encode(c_encodeType), indicators.encode(c_encodeType), startdate.encode(c_encodeType), enddate.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve26RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return c.__tryResolvePandas(data, result[1])

    @staticmethod
    def chmc(code, indicators, startdate=None, enddate=None, options="", *arga, **argb):
        """
        历史分钟K线(同步请求) //code只支持单个股票
        :param code: 东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param startdate:开始日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param enddate:截止日期。如无分隔符，则必须为8位数字。格式支持:YYYYMMDD YYYY/MM/DD YYYY/M/D YYYY-MM-DD YYYY-M-D
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        :return:EmQuantData
        """
        code = c.__toString(code)
        indicators = c.__toString(indicators)
        result = c.__resolveOption(options)
        options = result[0]
        data = c.EmQuantData()

        if(enddate == None):
            enddate = datetime.today().strftime("%Y-%m-%d")
        if(startdate == None):
            startdate = enddate
        if(isinstance(startdate, datetime) or isinstance(startdate, date)):
            startdate = startdate.strftime("%Y-%m-%d")
        if(isinstance(enddate, datetime) or isinstance(enddate, date)):
            enddate = enddate.strftime("%Y-%m-%d")
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_chmc(code.encode(c_encodeType), indicators.encode(c_encodeType), startdate.encode(c_encodeType), enddate.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve26RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return c.__tryResolvePandas(data, result[1])

    @staticmethod
    def csq(codes, indicators, options="", fncallback=None, userparams=None, *arga, **argb):
        """
        实时行情(异步)  每次indicators最多为64个 options: Pushtype=0 增量推送  1全量推送
        :param codes:东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param options:Pushtype=0 增量推送  1全量推送
        :param fncallback:不同的接口可以设定不同的回调，传NULL则使用默认的主回调函数
        :param userparams:用户参数,回调时原样返回
        :param arga:
        :param argb:
        :return:流水号
        """
        codes = c.__toString(codes)
        indicators = c.__toString(indicators)
        data = c.EmQuantData()
        global appQuoteCallback
        global appQuoteFunctionDict
        if not callable(fncallback):
            appQuoteFunctionDict[10000][0] = DemoCallback
        else:
            appQuoteFunctionDict[10000][0] = fncallback
            
        data.SerialID = c.quant_csq(codes.encode(c_encodeType), indicators.encode(c_encodeType), options.encode(c_encodeType), appQuoteCallback, userparams)

        if not callable(fncallback):
            appQuoteFunctionDict[10000][data.SerialID] = DemoCallback
        else:
            appQuoteFunctionDict[10000][data.SerialID] = fncallback
        return data

    @staticmethod
    def csqcancel(serialID):
        """
        取消实时行情订阅
        :param serialID:
        :return:
        """
        data = c.EmQuantData()
        data.ErrorCode = c.quant_csqcancel(serialID)
        data.ErrorMsg = c.geterrstring(data.ErrorCode)
        return data

    @staticmethod
    def cst(codes, indicators, startdatetime, enddatetime, options = "", fncallback=None, userparams=None):
        '''
        日内跳价服务(异步)  startdatetime和enddatetime格式(YYYYMMDDHHMMSS或HHMMSS表示系统日期当天的时间，两者需使用同一种格式)
        :param codes:东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param startdate:开始时间
        :param enddate:结束时间
        :param options:
        :param fncallback:不同的接口可以设定不同的回调，传NULL则使用默认的主回调函数
        :param userparams:用户参数,回调时原样返回
        :param arga:
        :param argb:
        :return:流水号
        '''
        codes = c.__toString(codes)
        indicators = c.__toString(indicators)
        data = c.EmQuantData()
        global appQuoteCallback
        global appQuoteFunctionDict
        if not callable(fncallback):
            appQuoteFunctionDict[10001][0] = cstCallBack
        else:
            appQuoteFunctionDict[10001][0] = fncallback
            
        data.SerialID = c.quant_cst(codes.encode(c_encodeType), indicators.encode(c_encodeType), startdatetime.encode(c_encodeType), enddatetime.encode(c_encodeType),options.encode(c_encodeType), appQuoteCallback, userparams)

        if not callable(fncallback):
            appQuoteFunctionDict[10001][data.SerialID] = cstCallBack
        else:
            appQuoteFunctionDict[10001][data.SerialID] = fncallback
        return data

    @staticmethod
    def csqsnapshot(codes, indicators, options=""):
        '''
        行情快照(同步请求) 每次indicators最多为64个
        :param codes:东财代码  多个代码间用半角逗号隔开，支持大小写。如 "300059.SZ,000002.SZ,000003.SZ,000004.SZ"
        :param indicators:东财指标 多个指标间用半角逗号隔开，支持大小写。如 "open,close,high"
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        '''
        codes = c.__toString(codes)
        indicators = c.__toString(indicators)

        data = c.EmQuantData()
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_csqsnapshot(codes.encode(c_encodeType), indicators.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve25RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def ctr(ctrName, indicators="", options=""):
        '''
        获取专题报表(同步请求)
        :param ctrName:
        :param startdate:开始日期
        :param enddate:截止日期
        :param options:附加参数  多个参数以半角逗号隔开，"Period=1,Market=CNSESH,Order=1,Adjustflag=1,Curtype=1,Pricetype=1,Type=1"
        '''
        indicators = c.__toString(indicators)
        data = c.EmQuantData()

        eqData = stEQCtrData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_ctr(ctrName.encode(c_encodeType), indicators.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQCtrData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolveCtrData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def cps(cpsCodes, cpsIndicators, cpsConditions, cpsOptions=''):
        '''
        条件选股函数
        :param cpsCodes  代码
        :param cpsIndicators  指标
        :param cpsConditions  条件
        :param cpsOptions  附加参数
        '''
        cpsCodes = c.__toString(cpsCodes)
        cpsIndicators = c.__toString(cpsIndicators)
        cpsConditions = c.__toString(cpsConditions)

        data = c.EmQuantData()

        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_cps(cpsCodes.encode(c_encodeType), cpsIndicators.encode(c_encodeType), cpsConditions.encode(c_encodeType), cpsOptions.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve2RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def setserverlistdir(serlistpath):
        """
        设置serverlist.json函数
        :param serlistpath 文件的路径
        """
        c.quant_setserlistdir(serlistpath.encode(c_encodeType))

    @staticmethod
    def setproxy(type, proxyip, port, verify, usr, pwd):
        """
        设置代理函数
        :param type 代理类型
        :param proxyip 代理ip
        :param port 端口
        :param verify 是否验证
        :param usr 代理用户名
        :param pwd 密码
        """
        data = c.EmQuantData()
        coutResult = c.quant_setproxy(type, proxyip.encode(c_encodeType), port, verify, usr.encode(c_encodeType), pwd.encode(c_encodeType))
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        return data

    @staticmethod
    def manualactivate(uname, password, options="", logcallback = None):
        """
        手动激活函数
        :param uname 用户名
        :param propasswordxyip 密码
        """
        data = c.EmQuantData()
        loginInfo = stEQLoginInfo()
        loginInfo.userName = uname.encode(c_encodeType)
        loginInfo.password = password.encode(c_encodeType)

        appLogCallback = None
        if callable(logcallback):
            appLogCallback = c_LogCallback(logcallback)
        else:
            def log(logMessage):
                print("[EmQuantAPI Python]", logMessage.decode(c_encodeType))
                return 1
            appLogCallback = c_LogCallback(log)
        loginResult = c.quant_manualactivate(pointer(loginInfo), options.encode(c_encodeType), appLogCallback)

        if loginResult != 0:
            data.ErrorCode = loginResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        return data

    @staticmethod
    def pquery(options=""):
        '''
        查询函数
        :param options 可选参数
        '''
        data = c.EmQuantData()
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_pquery(options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve25RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def porder(combincode, orderdict, remark="",options=""):
        '''
        下单函数
        :param combincode 组合代码
        :param orderdict 下单参数
        :param remark 备注
        :param options 可选参数
        '''
        if not isinstance(orderdict, dict):
            return None
        num = len(orderdict["code"])
        size = sizeof(stOrderInfo)
        orderinfo = (stOrderInfo*num)()
        for index in range(0, num):
            if "code" in orderdict.keys():
                orderinfo[index].code = orderdict["code"][index].encode(c_encodeType)
            if "volume" in orderdict.keys():
                orderinfo[index].volume = orderdict["volume"][index]
            if "price" in orderdict.keys():
                orderinfo[index].price = orderdict["price"][index]
            if "date" in orderdict.keys():
                td = orderdict["date"][index].replace("-", "").replace("/", "")
                orderinfo[index].date = int(td)
            if "time" in orderdict.keys():
                tt = orderdict["time"][index].replace(":", "")
                orderinfo[index].time = int(tt)
            if "optype" in orderdict.keys():
                orderinfo[index].optype = orderdict["optype"][index]
            if "cost" in orderdict.keys():
                orderinfo[index].cost = orderdict["cost"][index]
            if "rate" in orderdict.keys():
                orderinfo[index].rate = orderdict["rate"][index]
            if "reserve" in orderdict.keys():
                orderinfo[index].rate = orderdict["reserve"][index]

        data = c.EmQuantData()

        coutResult = c.quant_porder(pointer(orderinfo[0]), num, combincode.encode(c_encodeType), remark.encode(c_encodeType), options.encode(c_encodeType))
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)

        return data

    @staticmethod
    def edb(edbids, options):
        '''
        宏观指标服务
        '''
        edbids = c.__toString(edbids)

        data = c.EmQuantData()
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_edb(edbids.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve3RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def edbquery(edbids, indicators="", options=""):
        '''
        宏观指标id详情查询
        '''
        edbids = c.__toString(edbids)
        indicators = c.__toString(indicators)

        data = c.EmQuantData()
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_edbquery(edbids.encode(c_encodeType), indicators.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve3RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data

    @staticmethod
    def pcreate(combinCode, combinName, initialFound, remark, options=""):
        '''
        新建组合
        '''
        combinCode = c.__toString(combinCode)

        data = c.EmQuantData()

        coutResult = c.quant_pcreate(combinCode.encode(c_encodeType), combinName.encode(c_encodeType), initialFound, remark.encode(c_encodeType), options.encode(c_encodeType))
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        return data

    @staticmethod
    def pdelete(combinCode, options=""):
        '''
        删除组合
        '''
        combinCode = c.__toString(combinCode)
        data = c.EmQuantData()

        coutResult = c.quant_pdelete(combinCode.encode(c_encodeType), options.encode(c_encodeType))
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        return data

    @staticmethod
    def preport(combinCode, indicator, options=""):
        '''
        组合报表查询
        '''
        combinCode = c.__toString(combinCode)
        data = c.EmQuantData()
        eqData = stEQData()
        refEqData = byref(pointer(eqData))
        coutResult = c.quant_preport(combinCode.encode(c_encodeType), indicator.encode(c_encodeType), options.encode(c_encodeType), refEqData)
        if coutResult != 0:
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            return data
        tempData = refEqData._obj.contents
        if not isinstance(tempData, stEQData):
            data.ErrorCode = coutResult
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
        else:
            data.resolve25RankData(tempData)
            c.quant_releasedata(pointer(tempData))
        return data


    @staticmethod
    def __ShowBlankOption(options=""):
        pos = options.lower().find("showblank")
        ShowBlank = None
        if(pos >= 0):
            info = options[pos:]
            pos = info.find(",")
            if(pos > 0):
                info = info[0:pos]
            snum = info.split("=")[1]
            if(snum.isdigit() or (snum.startswith('-') and snum[1:].isdigit())):
                ShowBlank = int(snum)
        return ShowBlank

    @staticmethod
    def __resolveOption(arg=""):
        # Ispandas=1，RowIndex=1
        result_list = []
        pdDict = {}
        up_str = arg.upper()
        pos = up_str.find("ISPANDAS")
        if(pos >= 0):
            item = up_str[pos:pos+10]
            pdDict["ISPANDAS"] = item[9:10]
            arg = arg[0:pos]+arg[pos+10:]
        else:
            pdDict["ISPANDAS"] = "0"
        up_str = arg.upper()
        pos = up_str.find("ROWINDEX")
        if(pos >= 0):
            item = up_str[pos:pos+10]
            pdDict[item[0:8]] = item[9:10]
            arg = arg[0:pos]+arg[pos+10:]
        else:
            pdDict["ROWINDEX"] = "1"
        result_list.append(arg)
        result_list.append(pdDict)
        return result_list


    @staticmethod
    def __tryResolvePandas(data, args={}, fun_name=None):
        if data.ErrorCode != 0:
            return data
        if not (args != None and len(args) > 0 and "ISPANDAS" in args.keys() and "ROWINDEX" in args.keys() and args["ISPANDAS"] == "1"):
            return data

        if fun_name == None:
            import inspect
            fun_name = inspect.stack()[1][3]

        import pandas as pd

        code_list = []
        date_list = []
        data_list = []
        indictor_list = ["CODES", "DATES"]
        for ind in data.Indicators:
            data_list.append([])
        indictor_list.extend(data.Indicators)
        
        if(fun_name == "csc" or fun_name == "cmc" or fun_name == "chmc"):
            for code_index in range(0, len(data.Codes)):
                code = data.Codes[code_index]
                date_list.extend(data.Dates)
                for nIndex in range(0, len(data.Dates)):
                    code_list.append(code)
                for cIndex in range(0, len(data.Data)):
                    data_list[cIndex].extend(data.Data[cIndex])
        elif(fun_name == "csd"):
            data.Dates = [datetime.strptime(it,"%Y/%m/%d").strftime("%Y/%m/%d") for it in data.Dates]
            for code_index in range(0, len(data.Codes)):
                code = data.Codes[code_index]
                date_list.extend(data.Dates)
                for nIndex in range(0, len(data.Dates)):
                    code_list.append(code)
                for cIndex in range(0, len(data.Data[code])):
                    data_list[cIndex].extend(data.Data[code][cIndex])
        elif(fun_name == "css"):
            for code_index in range(0, len(data.Codes)):
                code = data.Codes[code_index]
                date_list.extend(data.Dates)
                for nIndex in range(0, len(data.Dates)):
                    code_list.append(code)
                for cIndex in range(0, len(data.Data[code])):
                    data_list[cIndex].append(data.Data[code][cIndex])
        else:
            return data

        data_list.insert(0, date_list)
        data_list.insert(0, code_list)
        table = pd.DataFrame(data_list, indictor_list)
        table = table.T

        if(args["ROWINDEX"] == "1"):
            table = table.sort_values(by=["CODES","DATES"]).set_index(["CODES"])
        elif(args["ROWINDEX"] == "2"):
            # table = table.sort(["DATES"]).set_index(["DATES"])
            table = table.sort_values(by=["DATES","CODES"]).set_index(["DATES"])
        return table


    @staticmethod
    def __toStrArray(args):
        if(args==None or args == ""):
            return [""]
        if(isinstance(args, str)):
            return [args]
        if(isinstance(args, int) or isinstance(args, float)) :
            return [str(args)]
        if(isinstance(args, tuple) or isinstance(args, list)):
            result = []
            for item in args:
                result.extend(c.__toStrArray(item))
            return result
        return [str(args)]

    @staticmethod
    def __toNumArray(args):
        if(args == None or args == ""): return None
        if(isinstance(args, tuple)): return [int(x) for x in args]
        if(isinstance(args, list)): return [int(x) for x in args]
        if(isinstance(args, int)): return [args]
        return None

    @staticmethod
    def __toString(args, joinStr = ","):
        v = c.__toStrArray(args)
        if(v == None): return None
        return joinStr.join(v)


def QuoteCallback(quotemessage, userparams):
    """
    实时行情回调处理函数
    :param quotemessage:
    :param userparams:
    :return:
    """
    try:
        quoteReceiveData = quotemessage.contents
        global appQuoteFunctionDict
        quotecallbackhandle = None
        data =c.EmQuantData()
        data.SerialID = quoteReceiveData.serialID
        data.RequestID = quoteReceiveData.requestID
        if quoteReceiveData.msgType == 0 or quoteReceiveData.msgType == 3:
            data.ErrorCode = quoteReceiveData.err
            data.ErrorMsg = c.geterrstring(data.ErrorCode)
            quotecallbackhandle = appQuoteFunctionDict[quoteReceiveData.requestID].get(0)
        else:
            data.resolve25RankData(quoteReceiveData.pEQData[0])
            quotecallbackhandle = appQuoteFunctionDict[quoteReceiveData.requestID].get(quoteReceiveData.serialID)
        if not callable(quotecallbackhandle):
            quotecallbackhandle = DemoCallback
        quotecallbackhandle(data)
    except Exception as e:
        print("QuoteCallback Exception", e)
    return 1

def DemoCallback(quantdata):
    """
    DemoCallback 是EM_CSQ订阅时提供的回调函数模板。该函数只有一个为c.EmQuantData类型的参数quantdata
    :param quantdata:c.EmQuantData
    :return:
    """
    print("QuoteCallback,", str(quantdata))

def cstCallBack(quantdata):
    '''
    cstCallBack 是日内跳价服务提供的回调函数模板
    '''
    for i in range(0, len(quantdata.Codes)):
        length = len(quantdata.Dates)
        for it in quantdata.Data.keys():
            print(it)
            for k in range(0, length):
                for j in range(0, len(quantdata.Indicators)):
                    print(quantdata.Data[it][j * length + k], " ", end = "")
                print()


appQuoteCallback = c_DataCallback(QuoteCallback)
