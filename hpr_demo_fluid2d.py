import sys
import csv
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import random

'''
Function used to Import csv Points
'''
def importPoints(fileName):

	p = np.empty(shape = [0,2]) # Initialize points p
	for line in open(fileName): #For reading lines from a file, loop over the file object. Memory efficient, fast, and simple:
		line = line.strip('\n') # Get rid of tailing \n
		line = line.strip('\r') # Get rid of tailing \r
		x,y,z = line.split(",") # In String Format
		p = np.append(p, [[float(x),float(z)]], axis = 0) 

	return p


def sphericalFlip(points, center, param):

	n = len(points) # total n points
	points = points - np.repeat(center, n, axis = 0) # Move C to the origin
	normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
	#R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere
	R = np.repeat(max(normPoints) * np.power(2.0, param), n, axis = 0) # Radius of Sphere
	flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
	flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
	flippedPoints += points 

	return flippedPoints

def ExponentialFlip(points, center, param):

	n = len(points) # total n points
	points = points - np.repeat(center, n, axis = 0) # Move C to the origin
	normPoints = np.linalg.norm(points, axis = 1) # Normed points, 
	#print(normPoints)
	normPointsScale = normPoints * 1.3
	flippedPoints = np.divide(points, np.repeat(normPointsScale.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
	flippedPoints += points 

	return flippedPoints


'''
Function used to Obtain the Convex hull
'''
def convexHull(points,c):

	points = np.append(points, c, axis = 0) 
	hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.

	return hull


'''
Main Function:
Apply Hidden Points Removal Operator to the Given Point Cloud
'''
def Main():
	myPoints = importPoints('points.csv', 3) # Import the Given Point Cloud

	C = np.array([[0,0,100]]) # View Point, which is well above the point cloud in z direction
	flippedPoints = sphericalFlip(myPoints, C, math.pi) # Reflect the point cloud about a sphere centered at C
	myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud


	# Plot
	fig = plt.figure(figsize = plt.figaspect(0.5))
	plt.title('Cloud Points With All Points (Left) vs. Visible Points Viewed from Well Above (Right)')
	
	# First subplot
	ax = fig.add_subplot(1,2,1, projection = '3d')
	ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	# Second subplot
	ax = fig.add_subplot(1,2,2, projection = '3d')
	for vertex in myHull.vertices[:-1]: # Exclude Origin, which is the last element
		ax.scatter(myPoints[vertex, 0], myPoints[vertex, 1], myPoints[vertex, 2], c='b', marker='o') # Plot visible points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()


def VisiblePointsAll(filename):
	xmyPoints = importPoints(filename) 
	randomRows = np.random.randint(len(xmyPoints), size=2001)
	myPoints=np.array([ xmyPoints[i,:]  for i in randomRows])

	############################ Method 1: Use a flag array indicating visibility, most efficent in speed and memory ############################
	flag = np.zeros(len(myPoints)+1, int) # Initialize the points
	flage = np.zeros(len(myPoints)+1, int) # Initialize the points visible from possible 6 locations. 0 - Invisible; 1 - Visible.
	C = np.array([[[0,0.0]], [[1.6,0.0]],[[0,0.5]], [[1.6,0.5]]])  # List of Centers
	for c in C:
		flippedPoints = sphericalFlip(myPoints, c, math.pi)
		flippedPointse = ExponentialFlip(myPoints, c, math.pi)
		print(len(flippedPoints),len(flippedPointse))
		myHull = convexHull(flippedPoints,c)
		myHulle = convexHull(flippedPointse,c)
		print(len(myHull.vertices),len(myHulle.vertices))
		visibleVertex = myHull.vertices[:-1] 
		visibleVertexe = myHulle.vertices[:-1] 
		flag[visibleVertex] = 1
		flage[visibleVertexe] = 1

	invisibleId = np.where(flag == 1)[0] # indexes of visible points
	invisibleIde =np.where(flage == 1)[0] # indexes of visible points

	# Plot for method 1
	fig = plt.figure(figsize = (12,6),dpi=70)
	#plt.title('HPR')
	
	# First subplot
	ax1 = fig.add_subplot(1,3,1)
	ax1.scatter(myPoints[:, 0], myPoints[:, 1],  c='r', marker='.') # Plot all points
	for c in C:
	   ax1.scatter(c[0,0], c[0,1], c='g', marker='v')

	ax1.set_xlabel('X Axis')
	ax1.set_ylabel('Y Axis')
	#ax1.set_zlabel('Z Axis')
	ax1.title.set_text('Cloud Points')
	#ax1.view_init(30., 30.) 
	# Second subplot
	ax2 = fig.add_subplot(1,3,2)
	for i in invisibleId:
	   if i<2001:
	     ax2.scatter(myPoints[i, 0], myPoints[i, 1], c='b', marker='.') # Plot visible points
	for c in C:
	   ax2.scatter(c[0,0], c[0,1], c='g', marker='v')
	ax2.set_xlabel('X Axis')
	ax2.set_ylabel('Y Axis')
	#ax2.set_zlabel('Z Axis')
	ax2.title.set_text('Spherical flipping')
	#ax.view_init(30., 30.) 
	# Thrird subplot
	ax3 = fig.add_subplot(1,3,3)
	for i in invisibleIde:
	   if i<2001:
	     ax3.scatter(myPoints[i, 0], myPoints[i, 1], c='b', marker='.') # Plot visible points
	for c in C:
	   ax3.scatter(c[0,0], c[0,1], c='g', marker='v')
	ax3.set_xlabel('X Axis')
	ax3.set_ylabel('Y Axis')
	#ax3.set_zlabel('Z Axis')
	ax3.title.set_text('Exponential flipping')
	#ax3.view_init(30., 30.) 
	#fig.axis('off')
	#fig.axes.get_xaxis().set_visible(False)
	#fig.axes.get_yaxis().set_visible(False)
# rotate the axes and update
	#for angle in range(0, 360):
	#  ax1.view_init(30, angle)
	#  ax2.view_init(30, angle)
	#  ax3.view_init(30, angle)
	#  plt.draw()
	#  plt.savefig('pict{0}.png'.format((angle)), bbox_inches='tight', pad_inches = 0)
	#  plt.pause(.001)

	plt.show()
	return

'''
Execution of Codes
'''
# Main()
# Test() 
#InvisiblePoints()
#VisiblePoints(sys.argv[1])
VisiblePointsAll(sys.argv[1])

