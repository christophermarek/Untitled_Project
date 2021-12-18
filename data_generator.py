
import random
import os
import py_vollib
from py_vollib.black_scholes_merton import black_scholes_merton

class DataGenerator():

    def __init__(self):

        self.moneyness = [0.8, 1.2] 
        self.time_to_maturity = [0.014, 1]
        self.dividend_rate = [0.0, 0.10]
        self.annualized_interest_rate = [0.00, 0.1]
        self.volatility = [0.1, 0.4]
        self.job_completed = False

    def updateMoneyness(self, updatedMoneyness):
        self.moneyness = updatedMoneyness

    def updateTimeToMaturity(self, updatedTimeToMaturity):
        self.time_to_maturity = updatedTimeToMaturity

    def updateDividendRate(self, updatedDividendRate):
        self.dividend_rate = updatedDividendRate

    def updateInterestRate(self, updatedInterestRate):
        self.annualized_interest_rate = updatedInterestRate
    
    def updateVolatility(self, updatedVolatility):
        self.volatility = updatedVolatility

    def getJobCompleted(self):
        return self.job_completed

    # creates a csv file containing a simulated dataset
    # params
    # size(int) = size of dataset to generate
    # destination(string) = where the file will be saved
    # Returns messages as a list [isError: bool, outputMessage: string]
    def generateDataSet(self, size, destination):
        if not isinstance(size, int):
            return [False, "Invalid size paramater passed"]

        # Prob should further check if its a destination string
        if not isinstance(destination, str):
            return [False, "Invalid destination paramater passed"]

        # Create output folder
        os.makedirs("output", exist_ok=True)

        # will reset file every time it runs
        file = open("output/" + destination, "w")
        file.write("moneyness,timetomaturity,dividendrate,interestrate,volatility,BS-Call\n")

        for i in range(size):
            # we round to maintain decimal place
            genMoneyness = round(random.uniform(self.moneyness[0], self.moneyness[1]), 1)
            genMaturity = round(random.uniform(self.time_to_maturity[0], self.time_to_maturity[1]), 2)
            genDividends = round(random.uniform(self.dividend_rate[0], self.dividend_rate[1]), 2)
            genInterest = round(random.uniform(self.annualized_interest_rate[0], self.annualized_interest_rate[1]), 2)
            genVolatility = round(random.uniform(self.volatility[0], self.volatility[1]), 2)
            # Assume stock price is 1 when reversing moneyness ratio, then everything is based on contract price
            underlyingPrice = 1.0
            strike = underlyingPrice / genMoneyness

            # price = black_scholes_merton(flag, S, K, t, r, sigma, q)
            bsCall = black_scholes_merton('c', strike, underlyingPrice, genMaturity, genInterest, genVolatility, genDividends)

            file.write(str(genMoneyness) + "," + str(genMaturity) +
                       "," + str(genDividends) + "," + str(genInterest) + "," + str(genVolatility) +"," + str(bsCall) +"\n")

        file.close()

        return [True, "Task Completed Successfully"]
