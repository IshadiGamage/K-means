#Importing the required libraries
import pandas as pd  				# Provides DataFrame Object for data manipulation
import numpy as np 					# For fast mathematical functions
from sklearn.cluster import KMeans  # Import K means algorithms from sklearn library
from sklearn import cluster
import matplotlib.pyplot as plt     # Matpltlib for plotting the graph

"----------------------------------------------------------------------------------------------------------"
									# Data cleansing
"----------------------------------------------------------------------------------------------------------"

#link address to download the dataset
#https://www.kaggle.com/aljarah/xAPI-Edu-Data/data

data = pd.read_csv("xAPI-Edu-Data.csv")

# Print the headers
print(list(data))

# View the basic structure of data
print(data.head())

# Print the total no of rows
print("Total number of rows: {0}".format(len(data)))

# ------------- Dealing with missing values --------------

# Replace NaN with an empty string or some other default value.
data.VisITedResources = data.VisITedResources.fillna("")
data.raisedhands = data.raisedhands.fillna("")

# Remove incomplete rows
data.dropna()

# count the number of Null values in each column
print("Count the total number of Null values in each column")
print("-----------------------------------------------------")
print(data.isnull().sum())

# Remove all the rows which has all NA values
data.dropna(how="all")

# No of rows and columns in the dataset after dropping the rows which contain the missing values
print("Total no: of rows and columns after dropping the rows which contain missing values ", data.shape)

# This tells Pandas that the column VNo of times visited Resources and No of raised hands needs to be a integer value
data = pd.read_csv("xAPI-Edu-Data.csv", dtype={"VisITedResources": int})
data = pd.read_csv("xAPI-Edu-Data.csv", dtype={"raisedhands": int})

# Rename columns VisITedResources as Visited_Resources and raisedhands as Raised_Hands
data.rename(columns = {'VisITedResources':'Visited_Resources', 'raisedhands':'Raised_Hands'})

data = data.rename(columns = {'VisITedResources':'Visited_Resources', 'raisedhands':'Raised_Hands'})

# Save results to a new csv file
data.to_csv('cleanfile.csv')

dataSet = pd.read_csv('cleanfile.csv')
print(dataSet.shape)
dataSet.head()

# Required data is extracted to a csv file called output.csv based on the columns Visited_Resources and Raised_Hands
df = pd.read_csv('cleanfile.csv')
header = ["Visited_Resources", "Raised_Hands"]
print('A csv file called output.csv is generated')
df.to_csv('output.csv', columns = header, header=0)


"----------------------------------------------------------------------------------------------------------"
									# K - means implementation
"----------------------------------------------------------------------------------------------------------"

# Cleaned data file is imported
filename = "output.csv"

# No of times Visited_Resources are assigned to a variable called feature_1
feature_1 = np.genfromtxt("output.csv", usecols=[1], delimiter=',')

# No of timwa Raised_Hands are assigned to a variable called feature_2
feature_2 = np.genfromtxt("output.csv", usecols=[2], delimiter=',')

# feature_1 and feature_2 data is printed on the terminal
print feature_1
print feature_2

# Displaying the plot
plt.plot()

# X and Y limits of the plot
plt.xlim([0, 100])
plt.ylim([0, 100])

# Assigning X and Y labels
plt.xlabel('No of times Visited Resources')
plt.ylabel('No of times Raised Hands')

# Title of the plot
plt.title('Student performance plotted on a graph')

plt.scatter(feature_1, feature_2)

# Display the plot
plt.show()
 
# Create a new plot from data
plt.plot()
X = np.array(list(zip(feature_1, feature_2))).reshape(len(feature_1), 2)

# Colors for the input data plot
colors = ['b','g','r','m']			# b - Blue, g - Green, r - Red, m - Magenta


# Marker colors for the output data plot
markers = ['o', 'v','s','h']		# o - circle marker, v - triangle down marker, s - square marker, h - hexagon marker

 
# No of clusters that we need to cluster the students according to marks
K = 4
kmeans_model = KMeans(n_clusters=K).fit(X)
 
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(feature_1[i], feature_2[i], color=colors[l], marker=markers[l],ls='None')

    # X and Y limits of the plot is defined
    plt.xlim([0, 100])
    plt.ylim([0, 100])

	# Ttle of the graph
    plt.title('Students clustered according to their classroom performance')

    # Assigning X and Y variables
    plt.xlabel('No of times Visited Resources')
    plt.ylabel('No of times Raised Hands')

	
# Displaying the plot
plt.show()
