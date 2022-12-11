#Import plt from matplotlib
from turtle import color
import matplotlib.pyplot as plt


def ReadContext(filepath = 'log.txt'):
    #Define a tuple to store the data
    data = []
    #Open the log.txt file
    with open(filepath, 'r') as f:
        #Read the file line by line
        for line in f:
            #Split the line into a list
            line = line.split()
            #Append the list to the tuple
            data.append(line)
    #Get value of tuple with key
    return data

def GetAPValue(filepath = 'log.txt'):
    #Get the data from the log.txt file
    data = ReadContext(filepath)
    #Get the value of "train_lr" from the data
    mAP = [(x[-16]) for x in data]
    # Remove the symbol "," and "[" in the value of "mAP50"
    mAP = [x.replace(',', '') for x in mAP]
    mAP = [x.replace('[', '') for x in mAP]
    #Convert the value of "mAP50" to float
    mAP = [float(x) for x in mAP]
    # Get the mAP value
    mAP50 = [(x[-15]) for x in data]
    mAP50 = [x.replace(',', '') for x in mAP50]
    # Convert the value of "train_lr" from string to float
    mAP50 = [float(x) for x in mAP50]
    return mAP50, mAP

#main function
if __name__ == '__main__':
    #Read context from log.txt
    #data = ReadContext(filepath = 'outputs/log.txt')
    mAP50, mAP = GetAPValue(filepath = 'outputs_COLOR_1210/log.txt')
    # Save epoch, mAP50 and mAP to a csv file
    with open('outputs/plot.csv', 'w') as f:
        for i in range(len(mAP50)):
            f.write(str(i) + ',' + str(mAP50[i]) + ',' + str(mAP[i]) + '\r')

    figure = plt.figure(figsize=(10, 7), num= 1, clear=True)
    ax = figure.add_subplot()
    ax.plot(mAP50, label='mAP@0.5', color='blue')
    ax.plot(mAP, label='mAP@0.5:0.95', color='red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig('./mAP.png')
    plt.close("all")