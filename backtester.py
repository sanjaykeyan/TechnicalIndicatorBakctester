import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

class IterativeBase():
    def __init__(self, symbol ,start, end, interval, amount): #initialising the global variables for the class
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.initial_balance = amount
        self.current_balance = amount
        self.data = None
        self.perf = 0
        self.units = 0
        self.trades = 0
        self.position = 0
        self.buy_price = None
        self.sell_price = None
        self.stop = 0
        self.get_data()
        self.SMA_ret, self.MACD_ret, self.RSI_ret, self.BB_ret = 0, 0, 0, 0 

    def get_data(self): #reading the csv data 
        raw = yf.download(tickers = self.symbol, start = self.start, end = self.end, interval = self.interval).Close.to_frame()    
        raw["B&H returns"] = (raw.Close - raw.Close.shift(1))
        self.data = raw
        return raw

    def plot_data(self,df, cols = None): #function to plot data  
        if cols is None:
            cols = ["price","B&H returns"]
        fig = px.line(df, y=cols, title=self.symbol)
        fig.show()
    
    def get_values(self, bar): #to get the date and price at a particular row in the dataframe
        date = str(self.data.index[bar].date())
        price = round(self.data.Close.iloc[bar], 5)
        return date, price
    
    def print_current_balance(self, bar): #to print the current available balance after any strategy has been run
        date, price = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, b_price, units = None, amount = None): #function to execute buy order in backtesting
        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / b_price)
        self.current_balance -= units * b_price # reduce cash balance by "purchase price"
        self.units += units #increase no of units
        self.trades += 1 #check on the no of trades, to calculate trading costs
        print("{} |  Buying {} for {}".format(date, units, round(b_price, 5))) 
     
    def sell_instrument(self, bar, s_price, units = None, amount = None):
        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / s_price)
        self.current_balance += units * s_price # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        print("{} |  Selling {} for {}".format(date, units, round(s_price, 5)))
    
    def print_current_position_value(self, bar):
        date, price = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
    
    def stoploss_hit_sell(self, bar, units = None, amount = None): #stoploss has been hit in a long position
        date, price = self.get_values(bar)
        if amount is not None:
            units = int(amount/price)
        self.current_balance += units*price
        self.units -=units
        self.trades += 1
        print("{} |  Stoploss hit Selling {} for {}".format(date, units, round(price, 5)))
        
    def stoploss_hit_buy(self, bar,units = None, amount = None):
        date, price = self.get_values(bar)
        if amount is not None:
            units = int(amount/price)
        self.current_balance -= units*price
        self.units +=units
        self.trades += 1
        print("{} |  Stoploss hit Buying {} for {}".format(date, units, round(price, 5))) #stoploss has been hit in a short position
        
    def close_pos(self, bar): #close all positions
        date, price = self.get_values(bar)
        b_h_ret = (self.data.Close[-1] - self.data.Close[0]) * 100 / self.data.Close[0] #calculate the buy and hold returns 
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100 #the perfomance of any strategy
        self.perf = perf 
        self.print_current_balance(bar)
        print("{} | B&H Returns (%) = {}".format(date,b_h_ret))       
        print("{} | net performance (%) = {}".format(date, round(perf,2)))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")

class IterativeBacktest(IterativeBase):
    def go_long(self,bar,b_price,units = None,amount = None): 
        if self.position == -1: #if it is a short position first we have to go neutral
            self.buy_instrument(bar,b_price,units = -self.units)
        if units is not None: #after in neutral position again execute buy order
            self.buy_instrument(bar,b_price,units =units)
        elif amount is not None:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar,b_price,amount=amount)
    def go_short(self,bar,s_price,units = None,amount = None):
        if self.position == 1:
            self.sell_instrument(bar,s_price,units = self.units) #if we are in long position go neutral 
        if units is not None: #once neutral again execute sell order
            self.sell_instrument(bar,s_price,units = units)
        elif amount is not None:
            if amount == "all":
                amount = self.current_balance 
            self.sell_instrument(bar,s_price,amount = amount)
    def go_neutral(self,bar,price,units = None,amount = None):
        if self.position == -1: #if it is a short position we have to go neutral
            self.buy_instrument(bar,price,units = -self.units)
        elif self.position == 1: #if it is a long position we have to go neutral 
            self.sell_instrument(bar,price,units = self.units)
    
    def stoploss_check(self, bar, stoploss, position, buy_price = None, sell_price = None): #to chek if stoploss has been hit 
        if buy_price is not None:
                if ((((self.data.Close[bar])-(buy_price)) * 100 /(buy_price))<=-stoploss) and self.position == 1:
                    return True
        if sell_price is not None:
                if (((self.data.Close[bar])-(sell_price)) * 100 /(sell_price))>=stoploss and self.position == -1:
                    return True
    def stoploss_execute(self, bar, stoploss_percent, position, buy_price, sell_price): #execute sell and buy accordingly once stoploss is hit 
        if buy_price is not None:
                if self.stoploss_check(bar = bar, buy_price = self.buy_price, sell_price = None, stoploss = stoploss_percent, position = self.position):
                    self.stoploss_hit_sell(bar,units = self.units)
                    self.position=0
                    self.buy_price = None
                    self.stop = 1
        if sell_price is not None:
                if self.stoploss_check(bar = bar, buy_price = None, sell_price = self.sell_price, stoploss = stoploss_percent, position = self.position):
                    self.stoploss_hit_buy(bar, units = -self.units)
                    self.position=0
                    self.sell_price = None
                    self.stop = -1       

class strategy_tester(IterativeBacktest):
    def test_sma_strategy(self, SMA_S =5, SMA_L=20, stoploss_percent = 1):
        print(75*"-")
        print("Testing SMA strategy | {} | SMA_S = {} | SMA_L = {}".format(self.symbol,SMA_S,SMA_L))
        print(75*"-")
        
        #reset
        self.position = 0 
        self.trades = 0 
        self.buy_price = None
        self.sell_price = None
        self.stop = 0 
        self.perf = 0
        self.current_balance = self.initial_balance

        #prepare data
        dSMA = self.data
        dSMA["SMA_S"]=self.data.Close.rolling(SMA_S).mean() #creating short and long SMA's
        dSMA["SMA_L"]=self.data.Close.rolling(SMA_L).mean() 
        dSMA.dropna
        #sma crossover strategy
        for bar in range(len(dSMA)-1): #iterating through the dataframe  
            self.stoploss_execute(bar,stoploss_percent,self.position,self.buy_price,self.sell_price) #stoploss checking 
            #checking for SMA crossover and execute long and short positions.
            if dSMA["SMA_S"].iloc[bar]>dSMA["SMA_L"].iloc[bar]: 
                if self.position in [0,-1] and self.stop in [0,-1]:
                    self.go_long(bar,dSMA["Close"].iloc[bar],amount ="all")
                    self.buy_price = dSMA.Close[bar]
                    self.stop = 0
                    self.position = 1
            elif dSMA["SMA_S"].iloc[bar]<dSMA["SMA_L"].iloc[bar]:
                if self.position in [0,1] and (self.stop ==0 or self.stop ==1):
                    self.go_short(bar,dSMA["Close"].iloc[bar], amount ="all")
                    self.sell_price = dSMA.Close[bar]
                    self.stop = 0
                    self.position = -1
        self.close_pos(bar+1)
        self.SMA_ret = self.perf    
    def test_MACD_strategy(self ,stoploss_percent = 2):
        print(75*"-")
        print("Testing MACD strategy | {} ".format(self.symbol))
        print(75*"-")
        
        #reset
        self.position = 0 
        self.trades = 0 
        self.buy_price = None
        self.sell_price = None
        self.stop = 0 
        self.perf = 0
        self.current_balance = self.initial_balance
        self.get_data()
       
        #prepare data
        dMACD = self.data
        ema12=self.data["Close"].ewm(span=12, adjust=False, min_periods=12).mean() # 12 period Exponential moving average 
        ema26=self.data["Close"].ewm(span=26, adjust=False, min_periods=26).mean() #26 period Exponential moving average 
        dMACD["MACD"] = ema12-ema26
        dMACD["MACD_s"] = dMACD.MACD.ewm(span=9, adjust=False, min_periods=9).mean()
        dMACD.dropna
        
        #strategy
        for bar in range(len(dMACD)-1):
            self.stoploss_execute(bar,stoploss_percent,self.position,self.buy_price,self.sell_price) 
            if dMACD["MACD"].iloc[bar]>dMACD["MACD_s"].iloc[bar]: #checking for MACD crossover
                if self.position in [0,-1] and self.stop in [0,-1]:
                    self.go_long(bar,dMACD.Close.iloc[bar],amount ="all")
                    self.buy_price = dMACD.Close[bar]
                    self.stop = 0
                    self.position = 1 
                    
            elif dMACD["MACD"].iloc[bar]<dMACD["MACD_s"].iloc[bar]:
                if self.position in [0,1] and (self.stop ==0 or self.stop ==1):
                    self.go_short(bar, dMACD.Close.iloc[bar],amount ="all")
                    self.sell_price = dMACD.Close[bar]
                    self.stop = 0
                    self.position = -1
        self.close_pos(bar+1)
        self.MACD_ret = self.perf

    def test_RSI_strategy(self, period = 14, stoploss_percent = 3):
        print(75*"-")
        print("Testing RSI Strategy | {} ".format(self.symbol))
        print(75*"-")
        
        #reset
        self.position = 0 
        self.trades = 0 
        self.buy_price = None
        self.sell_price = None
        self.stop = 0 
        self.perf = 0
        self.current_balance = self.initial_balance
        
        #prepare data
        dRSI = self.data 
        delta = dRSI["Adj Close"].diff(1) #taking the difference of each days close with the previous days close 
        #for the RSI indicator we need the average gain and average loss
        #Hence splitting the delta into positive and negative groups
        positive = delta.copy()
        negative = delta.copy()
        positive[positive < 0] = 0
        negative[negative > 0] = 0
        dRSI["avg_gain"] = positive.rolling(window = period).mean() 
        dRSI["avg_loss"] = abs(negative.rolling(window = period).mean())
        RSI = 100 - (100/(1+(dRSI.avg_gain/dRSI.avg_loss)))
        dRSI["RSI"] = dRSI.index.map(RSI) 
        
        #strategy
        for bar in range(len(dRSI)-1):
            self.stoploss_execute(bar,stoploss_percent,self.position,self.buy_price,self.sell_price) 
            if dRSI["RSI"].iloc[bar]<30: #when the RSI value goes below 30 execute buy order
                if self.position in [0,-1] and self.stop in [0,-1]:
                    self.go_long(bar,dRSI.Close[bar], amount ="all")
                    self.buy_price = dRSI.Close[bar]
                    self.stop = 0
                    self.position = 1 
                    
            elif dRSI["RSI"].iloc[bar]>70: #when the RSI value goes above 70 execute sell order
                if self.position in [0,1] and (self.stop ==0 or self.stop ==1):
                    self.go_short(bar, dRSI.Close[bar], amount ="all")
                    self.sell_price = dRSI.Close[bar]
                    self.stop = 0
                    self.position = -1
        self.close_pos(bar+1)
        self.RSI_ret = self.perf
    def test_bollingerband_strategy(self, SMA = 20, dev = 2, stoploss_percent = 3):
        print(75*"-")
        print("Testing Bollinger Band strategy | {}".format(self.symbol))
        print(75*"-")
        
        #reset
        self.position = 0 
        self.trades = 0 
        self.buy_price = None
        self.sell_price = None
        self.stop = 0 
        self.perf = 0
        self.current_balance = self.initial_balance
        
        #prepare data
        dBB = self.data
        dBB["SMA"] = dBB["Close"].rolling(SMA).mean()
        dBB["lower"]=dBB.SMA - dBB["Close"].rolling(SMA).std() * dev
        dBB["upper"]=dBB.SMA + dBB["Close"].rolling(SMA).std() * dev
        dBB["distance"] = dBB.Close - dBB.SMA
        dBB.dropna
        
        #strategy
        for bar in range(len(dBB)-1):
            self.stoploss_execute(bar,stoploss_percent,self.position,self.buy_price,self.sell_price)
            if bar > 0 : 
                if (dBB.distance.iloc[bar] * dBB.distance.iloc[bar-1]) < 0 and self.position == 1:
                    self.sell_instrument(bar,dBB["High"].iloc[bar], units = self.units)
                    self.position = 0
                elif (dBB.distance.iloc[bar] * dBB.distance.iloc[bar-1]) < 0 and self.position == -1:
                    self.buy_instrument(bar,dBB["Low"].iloc[bar], units = -self.units)
                    self.position = 0
            if dBB["Low"].iloc[bar] < dBB["lower"].iloc[bar]:
                if self.position in [0,-1] and self.stop in [0,-1]:
                    self.go_long(bar,dBB["Low"].iloc[bar],amount ="all")
                    self.buy_price = dBB.Low.iloc[bar]
                    self.stop = 0
                    self.position = 1          
            elif dBB["High"].iloc[bar] > dBB["upper"].iloc[bar]:
                if self.position in [0,1] and (self.stop ==0 or self.stop ==1):
                    self.go_short(bar,dBB["High"].iloc[bar], amount ="all")
                    self.sell_price = dBB["High"].iloc[bar]
                    self.stop = 0
                    self.position = -1
        self.close_pos(bar+1)
        self.BB_ret = self.perf  
    def test_combined_strategy(self, stoploss_percent = 2,SMA_S =5, SMA_L=20):
        #this strategy is based on SMA crossover and MACD crossover.
        #if SMA and MACD both gives buy signal then go long 
        #if SMA and MACD both gives sell signal then go short
        #this strategy has been chosen because in all the previous backtest results SMA and MACD are the top performing 
        #reset
        self.position = 0 
        self.trades = 0 
        self.buy_price = None
        self.sell_price = None
        self.stop = 0 
        self.perf = 0
        self.current_balance = self.initial_balance
        self.get_data()
        temp = self.data
        temp["SMA_S"]=temp.Close.rolling(SMA_S).mean()
        temp["SMA_L"]=temp.Close.rolling(SMA_L).mean()
        ema12=temp["Close"].ewm(span=12, adjust=False, min_periods=12).mean() # 12 period Exponential moving average 
        ema26=temp["Close"].ewm(span=26, adjust=False, min_periods=26).mean() #26 period Exponential moving average 
        temp["MACD"] = ema12-ema26
        temp["MACD_s"] = temp.MACD.ewm(span=9, adjust=False, min_periods=9).mean()
        temp.dropna
        for bar in range(len(temp)-1):
            self.stoploss_execute(bar,stoploss_percent,self.position,self.buy_price,self.sell_price) 
            if (temp["SMA_S"].iloc[bar]>temp["SMA_L"].iloc[bar]) and temp["MACD"].iloc[bar]>temp["MACD_s"].iloc[bar]:
                if self.position in [0,-1] and self.stop in [0,-1]:
                    self.go_long(bar,temp["Close"].iloc[bar],amount ="all")
                    self.buy_price = temp.Close[bar]
                    self.stop = 0
                    self.position = 1
            elif temp["SMA_S"].iloc[bar]<temp["SMA_L"].iloc[bar] and temp["MACD"].iloc[bar]<temp["MACD_s"].iloc[bar]:
                if self.position in [0,1] and (self.stop ==0 or self.stop ==1):
                    self.go_short(bar,temp["Close"].iloc[bar], amount ="all")
                    self.sell_price = temp.Close[bar]
                    self.stop = 0
                    self.position = -1
        self.close_pos(bar+1)
        self.get_MDD()     
    def get_MDD(self,SMA_S = 5, SMA_L = 20):
        temp = self.data
        temp["SMA_S"]=temp.Close.rolling(SMA_S).mean()
        temp["SMA_L"]=temp.Close.rolling(SMA_L).mean()
        ema12=temp["Close"].ewm(span=12, adjust=False, min_periods=12).mean() # 12 period Exponential moving average 
        ema26=temp["Close"].ewm(span=26, adjust=False, min_periods=26).mean() #26 period Exponential moving average 
        temp["MACD"] = ema12-ema26
        temp["MACD_s"] = temp.MACD.ewm(span=9, adjust=False, min_periods=9).mean()
        temp.dropna
        temp["SMApos"] = np.where((temp.SMA_S>temp.SMA_L), 1, -1)
        temp["MACDpos"] = np.where((temp["MACD"]>temp["MACD_s"]), 1, -1)
        temp["position"] = np.where(temp.SMApos == temp.MACDpos, temp.MACDpos, None)
        temp.position.ffill(inplace = True)
        units = self.initial_balance / temp.Close.iloc[0]
        temp["bhreturn"] = (temp.Close - temp.Close.shift(1))*units
        temp["strategyreturn"] = temp.bhreturn * temp.position
        temp["cum_bhreturn"] = temp.bhreturn.cumsum()
        temp["cum_strategyreturn"] = temp.strategyreturn.cumsum()
        temp.dropna
        
        temp["chngpos"] = np.where(temp.position *temp.position.shift(1) < 0 , 1 , 0)
        temp.iat[0,-1] = temp.position.iloc[0]
        MDD = 0
        MDD_percent = 0 
        price1 = temp.Close[0]
        position = temp.position[0]
        for bar in range(1,len(temp)-1):
            if temp.chngpos.iloc[bar] != 0 :
                price2 = temp.Close.iloc[bar-1]
                if (price1-price2) * position < MDD :
                    MDD = price1 - price2
                    MDD_percent = (price1 - price2) *100 / price1 
                price1 = price2
                position = temp.position[bar]
        print("Max drawdown% = ",MDD_percent)
        self.plot_data(df = temp,cols = ["cum_strategyreturn","cum_bhreturn"])



print("************************ STOCK BACKTESTER ************************")
symbol = input("Enter Stock Name/Symbol : ")
start = input("Enter Start date (YYYY-MM-DD) : ")
end = input("Enter End date (YYYY-MM-DD) : ")
interval = input("Enter Frequency of data : ")
amount = int(input("Enter amount : "))

x=0
tester = strategy_tester(symbol,start,end,interval,amount)
print("1.  SMA crossover strategy ")
print("2.  MACD crossover strategy ")
print("3.  RSI strategy")
print("4.  Bollinger Band Strategy ")
print("5.  MACD+SMA - Combined strategy")
print("-1. Exit")
x=int(input("Enter Option"))
if(x==1):
    p=input(("Do you want custom intervals for SMA (Default 5,20)(y/n)?"))
    if(p=='y'):
        SMA_S=int(input("Enter SMA short : "))
        SMA_L=int(input("Enter SMA Long : "))
        stoploss=int(input("Enter stoploss Percent : "))
        tester.test_sma_strategy(SMA_S,SMA_L,stoploss)

    else:
        stoploss=int(input("Enter stoploss Percent : "))
        tester.test_sma_strategy(stoploss_percent=stoploss)

elif(x==2):
    stoploss=int(input("Enter stoploss Percent : "))
    tester.test_MACD_strategy(stoploss_percent=stoploss)

elif(x==3):
    stoploss=int(input("Enter stoploss Percent : "))
    tester.test_RSI_strategy(stoploss_percent=stoploss)
elif(x==4):
    stoploss=int(input("Enter stoploss Percent : "))
    tester.test_bollingerband_strategy(stoploss_percent=stoploss)

elif(x==5):
    stoploss=int(input("Enter stoploss Percent : "))
    tester.test_combined_strategy(stoploss_percent=stoploss)
elif(x==-1):
    print("***************THANK YOU***************")

else :
    print("Give a valid input")

