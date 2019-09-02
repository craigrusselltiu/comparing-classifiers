import sys
import csv
import math
from statistics import mean


## Preprocessing

# Read command line arguments
training = sys.argv[1]
testing = sys.argv[2]
algorithm = sys.argv[3]

# Read the csv files into a list of lists
with open(training, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

with open(testing, 'r') as f:
    reader = csv.reader(f)
    examples = list(reader)

# Pre-calculate mean and standard deviation for both classes yes and no for all examples
# Formatted as: [[mean_yes, mean_no, sd_yes, sd_no], ...]
mean_sd = []
last = len(data[0])-1

# Calculate total yes and no
total_yes = 0
total_no = 0

for i in data:
    if (i[last] == "yes"):
        total_yes += 1
    else:
        total_no += 1

total = total_yes + total_no

# For every attribute
def calc_mean_sd():
    for i in range(last):
        mean_sd.append([0, 0, 0, 0])

        # Calculate mean
        yes = 0; no = 0
        for j in range(len(data)):
            classified = data[j][last]
            
            if (classified == "yes"):
                mean_sd[i][0] += float(data[j][i])
                yes += 1
            else:
                mean_sd[i][1] += float(data[j][i])
                no += 1

        mean_sd[i][0] = mean_sd[i][0]/yes
        mean_sd[i][1] = mean_sd[i][1]/no
            
        # Calculate standard deviation
        for j in range(len(data)):
            classified = data[j][last]
            
            if (classified == "yes"):
                mean_sd[i][2] += (float(data[j][i]) - mean_sd[i][0]) ** 2
            else:
                mean_sd[i][3] += (float(data[j][i]) - mean_sd[i][1]) ** 2

        mean_sd[i][2] = (mean_sd[i][2]/(yes-1)) ** 0.5
        mean_sd[i][3] = (mean_sd[i][3]/(no-1)) ** 0.5

calc_mean_sd()


## Functions

# Calculates euclidean distance of list vectors a and b
def euclidean(a, b):
    sum = 0

    if (len(a) != len(b)):
        print("Error: Vectors must have equal length to calculate Euclidean Distance!")
    else:
        for i in range(0, len(a)):
            sum += (float(b[i])-float(a[i])) ** 2

    return sum ** 0.5

# Calculates probability density function given mean, standard deviation and input
def pdf(mean, sd, x):
    a = 1 / (sd * ((2 * math.pi) ** 0.5))
    b = math.e ** ((-1 * ((x - mean) ** 2)) / (2 * (sd ** 2)))
    return a * b          

# Returns the class predicted for x by the k-nearest neighbors algorithm
def k_nearest_neighbors(k, x):

    # Create a list of the data and their euclidian distances from example
    neighbors = []
    for i in data:
        neighbors.append([euclidean(i[:-1], x), i])

    # Sort the list by the euclidean distance (ascending)
    neighbors.sort(key=lambda x: x[0])

    # Count classes of the k-nearest neighbors
    yes = 0
    for i in range(k):
        classified = neighbors[i][1][len(neighbors[i][1])-1]
        if (classified == "yes"):
            yes += 1
    no = k-yes

    # If there are equal number of yes or greater, classify as yes (otherwise no)
    if (yes >= no):
        return "yes"
    else:
        return "no"

# Returns predicted class for x by naive bayes algorithm
def naive_bayes(x):
    p_yes = 1
    p_no = 1

    for i in range(len(x)):
        p_yes *= pdf(mean_sd[i][0], mean_sd[i][2], float(x[i]))
        p_no *= pdf(mean_sd[i][1], mean_sd[i][3], float(x[i]))

    p_yes *= total_yes/total
    p_no *= total_no/total
    
    if (p_yes >= p_no):
        return "yes"
    else:
        return "no"


## Main

# Process correct algorithm requested

if (algorithm == "NB"):

    # Print the predicted class for all example data
    for i in examples:
        print(naive_bayes(i))
else:

    # Get k from command line argument, i.e. 'k'NN
    algorithm = int(algorithm[:-2])

    # Print the predicted class for all example data
    for i in examples:
        print(k_nearest_neighbors(algorithm, i))


## Evaluation
evaluate = False

if (evaluate):
    # Number of folds
    num_folds = 10
    folds = []

    # Yes and no split lists and iterators
    yes = []
    no = []
    yes_i = 0
    no_i = 0

    # Extra examples to be split among folds
    yes_extra = total_yes % num_folds
    no_extra = total_no % num_folds

    # Split data into yes and no
    for i in data:
        if (i[last] == "yes"):
            yes.append(i)
        else:
            no.append(i)

    # Create every fold
    for i in range(num_folds):
        folds.append([])

        # Spread yes and no examples equally among folds (stratification)
        num_no = total_no // num_folds
        if (no_extra > 0):
            num_no += 1
            no_extra -= 1

        for j in range(num_no):
            folds[i].append(no[no_i])
            no_i += 1

        num_yes = total_yes // num_folds
        if (yes_extra > 0):
            num_yes += 1
            yes_extra -= 1

        for j in range(num_yes):
            folds[i].append(yes[yes_i])
            yes_i += 1
        
    # Write the folds into a csv file
    with open('pima-folds.csv', 'w') as f:
        for i in range(num_folds):
            f.write("fold{}\n".format(i+1))
            for j in folds[i]:
                for x in j:
                    f.write("{}".format(x))
                    if (x != "yes" and x != "no"):
                        f.write(",")
                f.write("\n")
            f.write("\n")

    # Sum of different folds of each algorithm
    nb = []
    k1 = []
    k5 = []

    print("Calculating accuracy via {}-fold cross validation...".format(num_folds))

    # K-fold cross validation
    for i in range(num_folds):

        # Training fold is every fold except current one
        data = []
        for j in folds:
            if (folds[i] != j):
                data += j

        # The current one is now the testing fold
        examples = folds[i]

        # Initialise accuracies and count
        acc_nb = 0
        acc_k1 = 0
        acc_k5 = 0
        count = 0

        # Recalculate mean_sd
        mean_sd = []
        calc_mean_sd()

        # Count number of times each algorithm is correct
        for j in examples:
            if (naive_bayes(j[:-1]) == j[last]):
                acc_nb += 1
            if (k_nearest_neighbors(1, j[:-1]) == j[last]):
                acc_k1 += 1
            if (k_nearest_neighbors(5, j[:-1]) == j[last]):
                acc_k5 += 1
            count += 1

        acc_nb /= count
        acc_k1 /= count
        acc_k5 /= count

        nb.append(acc_nb)
        k1.append(acc_k1)
        k5.append(acc_k5)

    print("Naive Bayes accuracy over {} folds is: {}%".format(num_folds, mean(nb)*100))
    print("1-Nearest Neighbor accuracy over {} folds is: {}%".format(num_folds, mean(k1)*100))
    print("5-Nearest Neighbor accuracy over {} folds is: {}%".format(num_folds, mean(k5)*100))