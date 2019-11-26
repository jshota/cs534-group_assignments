import matplotlib.pyplot as plt
import csv

def training_plot():
    x = []
    y = []

    with open('result_train.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]))

    plt.plot(x,y)
    plt.xscale('log')
    plt.xlabel('Iteration Times')
    plt.ylabel('SSE')
    plt.title('SSE Curve - Training')
    plt.savefig('SSE Curve - Training')

def validation_plot():
    x = []
    y = []

    with open('result_val.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(float(row[1]))

    plt.plot(x,y)
    plt.xscale('log')
    plt.xlabel('Iteration Times')
    plt.ylabel('SSE')
    plt.title('SSE Curve - Validation')
    plt.savefig('SSE Curve - Validation')

training_plot()
plt.close()
validation_plot()