
import random

class DataGenerator():
    
    def __init__(self):
        self.moneyness = [0.4, 1.6]
        self.time_to_maturity = [0.01, 1.5]
        self.dividend_rate = [0.0, 0.03]
        self.annualized_interest_rate = [0.02, 0.3] 
        self.job_completed = False

    def updateMoneyness(self, updatedMoneyness):
        self.moneyness = updatedMoneyness
    
    def updateTimeToMaturity(self, updatedTimeToMaturity):
        self.time_to_maturity = updatedTimeToMaturity

    def updateDividendRate(self, updatedDividendRate):
        self.dividend_rate = updatedDividendRate

    def updateInterestRate(self, updatedInterestRate):
        self.annualized_interest_rate = updatedInterestRate

    def getJobCompleted(self):
        return self.job_completed

    # should not return since it will be so large but rather create the file
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
        
        #will reset file every time it runs
        file = open(destination, "w")
        file.write("moneyness,timetomaturity,dividendrate,interestrate\n")

        for i in range(size):
            #we round to maintain decimal place
            genMoneyness = round(random.uniform(self.moneyness[0], self.moneyness[1]), 1)
            genMaturity = round(random.uniform(self.time_to_maturity[0], self.time_to_maturity[1]), 2)
            genDividends = round(random.uniform(self.dividend_rate[0], self.dividend_rate[1]), 2)
            genInterest = round(random.uniform(self.annualized_interest_rate[0], self.annualized_interest_rate[1]), 2)

            file.write(str(genMoneyness) + "," + str(genMaturity) + "," + str(genDividends) + "," + str(genInterest) + "\n")
            
        file.close()

        return [True, "Task Completed Successfully"]